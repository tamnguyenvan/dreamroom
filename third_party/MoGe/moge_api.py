# ---
# canonical: modal serve deployments/modal/moge_api.py
# ---

# # MoGe-2 FastAPI endpoint
#
# Deploy MoGe-2 as a simple REST API on Modal.
# POST an image, get back a ZIP containing:
#   - output.glb      (3D textured mesh, camera-aligned)
#   - point_map.npy   (H x W x 3 metric points in output.glb coordinates)
#   - depth.png       (colorized depth map)
#   - normal.png      (surface normal map)
#   - metadata.json   (intrinsics, FOV, image dimensions)
#
# The codebase + 1.32 GB model weights are baked into the image.

import io
import json
import os
import sys
import zipfile

import modal
from fastapi import FastAPI, File, Form, UploadFile, Response
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO = "Ruicheng/moge-2-vitl-normal"
REPO_URL = "https://huggingface.co/spaces/Ruicheng/MoGe-2"

# ---------------------------------------------------------------------------
# Image  (bake code + weights)
# ---------------------------------------------------------------------------


def download_model_weights():
    from huggingface_hub import snapshot_download
    print(f"Downloading {MODEL_REPO} …")
    snapshot_download(repo_id=MODEL_REPO, repo_type="model")
    print("Done.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6",
                 "libgl1-mesa-glx", "libglib2.0-0", "libgomp1")
    .run_commands(f"cd /root && git clone {REPO_URL} repo")
    .pip_install("setuptools", "wheel")
    .run_commands("pip install --no-cache-dir -r /root/repo/requirements.txt")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(
        download_model_weights,
        secrets=[modal.Secret.from_name("huggingface")],
    )
)

app = modal.App(name="moge-2-api", image=image)

# ---------------------------------------------------------------------------
# Shared model loader
# ---------------------------------------------------------------------------


def load_model():
    import torch
    from moge.model import import_model_class_by_version
    model = (
        import_model_class_by_version("v2")
        .from_pretrained("Ruicheng/moge-2-vitl-normal")
        .cuda()
        .eval()
        .half()
    )
    return model


# ---------------------------------------------------------------------------
# Web endpoint
# ---------------------------------------------------------------------------


@app.cls(
    gpu="A10G",
    max_containers=1,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
class MogeService:
    """Snapshot-backed MoGe 2 web service."""

    @modal.enter()
    def load_for_snapshot(self):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        sys.path.insert(0, "/root/repo")

        import torch

        print("Loading MoGe-2 model for GPU snapshot ...")
        self.model = load_model()
        warmup = torch.zeros(
            (3, 64, 64), dtype=torch.float16, device="cuda")
        with torch.inference_mode():
            self.model.infer(
                warmup,
                apply_mask=True,
                resolution_level=0,
                use_fp16=True,
            )
        torch.cuda.synchronize()
        print("MoGe-2 ready for snapshot.")

    @modal.asgi_app(label="moge-2-api-web")
    def web(self):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        sys.path.insert(0, "/root/repo")

        import time
        import numpy as np
        import cv2
        import torch
        import trimesh
        import trimesh.visual
        from PIL import Image
        import utils3d
        from moge.utils.vis import colorize_depth, colorize_normal
        from moge.utils.geometry_numpy import depth_occlusion_edge_numpy

        model = self.model

        web_app = FastAPI()

        # -----------------------------------------------------------------------
        # Helper: embed a perspective camera into a GLB binary
        # -----------------------------------------------------------------------
        def _embed_camera_in_glb(glb_bytes: bytes, intrinsics, img_w: int, img_h: int) -> io.BytesIO:
            """
            Parse a binary GLB, inject a glTF camera node at origin looking along -Z
            with the perspective projection matching the MoGe-2 intrinsics, and return
            a new BytesIO with the modified GLB.
            """
            import struct
            import json as _json

            # --- Parse GLB header ---
            magic = struct.unpack_from("<I", glb_bytes, 0)[0]
            if magic != 0x46546C67:
                # Not a valid GLB; return unmodified
                return io.BytesIO(glb_bytes)

            # --- Find JSON chunk ---
            pos = 12
            json_data = None
            bin_data = None
            bin_offset = 0
            while pos < len(glb_bytes):
                chunk_len = struct.unpack_from("<I", glb_bytes, pos)[0]
                chunk_type = struct.unpack_from("<I", glb_bytes, pos + 4)[0]
                chunk_data = glb_bytes[pos + 8 : pos + 8 + chunk_len]
                if chunk_type == 0x4E4F534A:   # "JSON"
                    json_data = chunk_data
                elif chunk_type == 0x004E4942:  # "BIN\0"
                    bin_data = chunk_data
                    bin_offset = pos + 8
                pos += 8 + chunk_len

            if json_data is None:
                return io.BytesIO(glb_bytes)

            gltf = _json.loads(json_data.decode("utf-8"))

            # --- Compute camera projection ---
            fy = float(intrinsics[1, 1])
            yfov = 2.0 * np.arctan(0.5 / fy)
            aspect_ratio = img_w / img_h

            # --- Add camera if not present ---
            cameras = gltf.setdefault("cameras", [])
            camera_idx = len(cameras)
            cameras.append({
                "type": "perspective",
                "perspective": {
                    "yfov": yfov,
                    "aspectRatio": aspect_ratio,
                    "znear": 0.001,
                    "zfar": 1000.0,
                },
                "name": "input_view",
            })

            # --- Add camera node at origin looking along -Z (glTF default) ---
            nodes = gltf.setdefault("nodes", [])
            camera_node_idx = len(nodes)
            nodes.append({
                "camera": camera_idx,
                "name": "camera_input",
                # Position at origin (0,0,0), no rotation (looks along -Z by default)
            })

            # --- Re-pack JSON chunk ---
            json_str = _json.dumps(gltf, separators=(",", ":"), ensure_ascii=False)
            json_bytes = json_str.encode("utf-8")
            # Pad with spaces to maintain 4-byte alignment
            while len(json_bytes) % 4 != 0:
                json_bytes += b" "
            # Pad to at most 3 extra bytes and a trailing 0x20 (space) for alignment
            json_padded = json_bytes

            # --- Rebuild GLB ---
            new_json_chunk_len = len(json_padded)
            new_json_header = struct.pack("<II", new_json_chunk_len, 0x4E4F534A)

            new_bin_header = b""
            new_bin_start = 0
            if bin_data is not None:
                new_bin_start = 12 + 8 + new_json_chunk_len
                new_bin_header = struct.pack("<II", len(bin_data), 0x004E4942)

            new_total_len = new_bin_start + (8 + len(bin_data) if bin_data else 0)
            new_header = struct.pack("<I", 0x46546C67) + struct.pack("<II", 2, new_total_len)

            out = io.BytesIO()
            out.write(new_header)
            out.write(new_json_header)
            out.write(json_padded)
            if bin_data is not None:
                out.write(new_bin_header)
                out.write(bin_data)
            out.seek(0)
            return out

        @web_app.post("/predict")
        async def predict(
            file: UploadFile = File(...),
            include_mesh: bool = Form(True),
            include_debug: bool = Form(True),
        ):
            t0 = time.perf_counter()

            raw = await file.read()
            np_arr = np.frombuffer(raw, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                return Response("Invalid image", status_code=400)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            max_sz = 800
            if max(h, w) > max_sz:
                scale = max_sz / max(h, w)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                h, w = image.shape[:2]

            image_tensor = torch.tensor(image, dtype=torch.float16, device="cuda").permute(2, 0, 1) / 255
            with torch.inference_mode():
                output = model.infer(image_tensor, apply_mask=True, resolution_level=9, use_fp16=True)
            output = {k: v.cpu().numpy() for k, v in output.items()}

            points = output["points"]
            depth = output["depth"]
            mask = output["mask"]
            normal = output.get("normal", None)
            intrinsics = output.get("intrinsics", None)   # (3, 3) normalized camera intrinsics

            # ---- FOV from intrinsics ----
            fov_x_deg = None
            if intrinsics is not None:
                fx = intrinsics[0, 0]  # focal length normalized by image width
                fov_x_deg = float(np.rad2deg(2 * np.arctan(0.5 / fx)))

            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)

            depth_png = None
            normal_png = None
            if include_debug:
                depth_visualization = depth.copy()
                depth_visualization[~mask_cleaned] = np.nan
                depth_png = colorize_depth(depth_visualization)
                if normal is not None:
                    normal_visualization = normal.copy()
                    normal_visualization[~mask_cleaned] = 0
                    normal_png = colorize_normal(normal_visualization)

            coordinate_flip = np.array([1, -1, -1], dtype=np.float32)
            point_map = points.astype(np.float32, copy=True) * coordinate_flip
            point_map[~mask_cleaned] = np.nan
            point_map_buf = io.BytesIO()
            np.save(point_map_buf, point_map, allow_pickle=False)

            glb_buf = None
            if include_mesh:
                image_uv = utils3d.numpy.image_uv(width=w, height=h)
                image_float = image.astype(np.float32) / 255
                if normal is None:
                    faces, vertices, _, vertex_uvs = (
                        utils3d.numpy.image_mesh(
                            points,
                            image_float,
                            image_uv,
                            mask=mask_cleaned,
                            tri=True,
                        ))
                    vertex_normals = None
                else:
                    faces, vertices, _, vertex_uvs, vertex_normals = (
                        utils3d.numpy.image_mesh(
                            points,
                            image_float,
                            image_uv,
                            normal,
                            mask=mask_cleaned,
                            tri=True,
                        ))
                vertices = vertices * coordinate_flip
                vertex_uvs = (
                    vertex_uvs * np.array([1, -1], dtype=np.float32) +
                    np.array([0, 1], dtype=np.float32))
                if vertex_normals is not None:
                    vertex_normals = vertex_normals * coordinate_flip

                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_normals=(
                        vertex_normals if vertex_normals is not None else None),
                    visual=trimesh.visual.texture.TextureVisuals(
                        uv=vertex_uvs,
                        material=trimesh.visual.material.PBRMaterial(
                            baseColorTexture=Image.fromarray(image),
                            metallicFactor=0.5,
                            roughnessFactor=1.0,
                        ),
                    ),
                    process=False,
                )
                glb_buf = io.BytesIO()
                mesh.export(glb_buf, file_type="glb")
                glb_buf.seek(0)
                if intrinsics is not None:
                    glb_buf = _embed_camera_in_glb(
                        glb_buf.getvalue(), intrinsics, w, h)

            # zip
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(
                zip_buf, "w", zipfile.ZIP_STORED,
            ) as zf:
                if glb_buf is not None:
                    zf.writestr("output.glb", glb_buf.read())
                zf.writestr("point_map.npy", point_map_buf.getvalue())
                if depth_png is not None:
                    zf.writestr(
                        "depth.png", Image.fromarray(depth_png)._repr_png_())
                if normal_png is not None:
                    zf.writestr("normal.png", Image.fromarray(normal_png)._repr_png_())

                # ---- Camera metadata ----
                meta = {
                    "image_size": [w, h],
                    "point_map_coordinates": "output.glb camera coordinates (+X right, +Y up, -Z forward)",
                }
                if intrinsics is not None:
                    meta["intrinsics"] = intrinsics.tolist()
                    meta["fov_x_deg"] = fov_x_deg
                    # x values are normalized by width; y values by height.
                    meta["intrinsics_convention"] = (
                        "normalized by image width/height "
                        "(principal_point=0.5,0.5)")
                zf.writestr("metadata.json", json.dumps(meta, indent=2))
            zip_buf.seek(0)

            elapsed = time.perf_counter() - t0
            print(f"[{file.filename}] {w}x{h} → {elapsed*1000:.0f} ms")
            return StreamingResponse(
                zip_buf, media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=output.zip",
                    "Server-Timing": f"moge;dur={elapsed*1000:.1f}",
                },
            )

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        return web_app
