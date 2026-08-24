"""Modal-hosted FastAPI service for MoGe-3.

Develop with:
    modal serve moge_api.py

Deploy with:
    modal deploy moge_api.py
"""

import io
import json
import math
import struct
import sys
import time
import zipfile

import modal


MODEL_REPO = "Ruicheng/moge-3-vitg"
SOURCE_REPO = "https://github.com/microsoft/MoGe.git"
SOURCE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def download_model_weights() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_REPO} ...")
    snapshot_download(repo_id=MODEL_REPO, repo_type="model")
    print("Model download complete.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
    )
    .run_commands(
        f"git clone --filter=blob:none --no-checkout {SOURCE_REPO} /root/repo",
        f"git -C /root/repo checkout {SOURCE_REVISION}",
    )
    .pip_install("fastapi[standard]", "setuptools", "wheel")
    .run_commands(
        "python -m pip install --no-cache-dir -e /root/repo "
        "--index-url https://download.pytorch.org/whl/cu130 "
        "--extra-index-url https://pypi.org/simple"
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTHONPATH": "/root/repo",
        }
    )
    .run_function(download_model_weights)
)

app = modal.App(name="moge-3-api", image=image)


def load_model():
    from moge.model.v3 import MoGeModel

    return MoGeModel.from_pretrained(MODEL_REPO).cuda().eval()


def _embed_camera_in_glb(
    glb_bytes: bytes,
    intrinsics,
    image_width: int,
    image_height: int,
) -> bytes:
    """Attach the inferred input camera to the default glTF scene."""
    if len(glb_bytes) < 12:
        return glb_bytes

    magic, version, total_length = struct.unpack_from("<III", glb_bytes)
    if magic != 0x46546C67 or version != 2 or total_length != len(glb_bytes):
        return glb_bytes

    chunks: list[tuple[int, bytes]] = []
    json_index = None
    position = 12
    while position + 8 <= len(glb_bytes):
        chunk_length, chunk_type = struct.unpack_from("<II", glb_bytes, position)
        chunk_end = position + 8 + chunk_length
        if chunk_end > len(glb_bytes):
            return glb_bytes
        if chunk_type == 0x4E4F534A:
            json_index = len(chunks)
        chunks.append((chunk_type, glb_bytes[position + 8 : chunk_end]))
        position = chunk_end

    if position != len(glb_bytes) or json_index is None:
        return glb_bytes

    try:
        gltf = json.loads(chunks[json_index][1].rstrip(b" \t\r\n\0"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return glb_bytes

    fy = float(intrinsics[1, 1])
    camera_index = len(gltf.setdefault("cameras", []))
    gltf["cameras"].append(
        {
            "name": "input_view",
            "type": "perspective",
            "perspective": {
                "aspectRatio": image_width / image_height,
                "yfov": 2.0 * math.atan(0.5 / fy),
                "znear": 0.001,
                "zfar": 1000.0,
            },
        }
    )

    camera_node_index = len(gltf.setdefault("nodes", []))
    gltf["nodes"].append(
        {
            "camera": camera_index,
            "name": "camera_input",
        }
    )

    scenes = gltf.setdefault("scenes", [{"nodes": []}])
    scene_index = gltf.get("scene", 0)
    if not isinstance(scene_index, int) or not 0 <= scene_index < len(scenes):
        scene_index = 0
        gltf["scene"] = scene_index
    scenes[scene_index].setdefault("nodes", []).append(camera_node_index)

    json_chunk = json.dumps(
        gltf,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    chunks[json_index] = (0x4E4F534A, json_chunk)

    body = b"".join(
        struct.pack("<II", len(chunk), chunk_type) + chunk
        for chunk_type, chunk in chunks
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def _png_bytes(image) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def _mesh_glb(
    image,
    points,
    normal,
    mask,
    intrinsics,
) -> bytes:
    import numpy as np
    import trimesh
    import trimesh.visual
    from PIL import Image

    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d

    height, width = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0
    uv_map = utils3d.np.uv_map(height, width)

    if normal is None:
        faces, vertices, _, vertex_uvs = utils3d.np.build_mesh_from_map(
            points,
            image_float,
            uv_map,
            mask=mask,
            tri=True,
        )
        vertex_normals = None
    else:
        faces, vertices, _, vertex_uvs, vertex_normals = (
            utils3d.np.build_mesh_from_map(
                points,
                image_float,
                uv_map,
                normal,
                mask=mask,
                tri=True,
            )
        )

    coordinate_flip = np.array([1, -1, -1], dtype=np.float32)
    vertices = vertices * coordinate_flip
    vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32)
    vertex_uvs += np.array([0, 1], dtype=np.float32)
    if vertex_normals is not None:
        vertex_normals = vertex_normals * coordinate_flip

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
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
    glb_bytes = mesh.export(file_type="glb")
    return _embed_camera_in_glb(glb_bytes, intrinsics, width, height)


@app.cls(
    gpu="H100",
    max_containers=1,
    scaledown_window=300,
    timeout=30 * 60,
)
class MogeService:
    """MoGe-3 ViT-G web service."""

    @modal.enter()
    def load(self) -> None:
        sys.path.insert(0, "/root/repo")

        import torch

        print("Loading MoGe-3 model ...")
        self.model = load_model()

        # Compile the v3 backbone and sparse-refinement paths before serving.
        warmup = torch.zeros((3, 64, 64), dtype=torch.float32, device="cuda")
        self.model.infer(
            warmup,
            apply_mask=True,
            refine_steps=1,
            resolution_level=0,
            use_fp16=True,
        )
        torch.cuda.synchronize()
        print("MoGe-3 ready.")

    @modal.asgi_app(label="moge-3-api-web")
    def web(self):
        import cv2
        import numpy as np
        import torch
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import StreamingResponse
        from moge.utils.vis import colorize_depth, colorize_normal

        try:
            import utils3d_moge as utils3d
        except ImportError:
            import utils3d

        web_app = FastAPI(title="MoGe-3 API", version="1.0.0")

        @web_app.post("/predict")
        async def predict(
            file: UploadFile = File(...),
            include_mesh: bool = Form(True),
            include_debug: bool = Form(True),
            max_size: int = Form(800, ge=64, le=2048),
            resolution_level: int = Form(9, ge=0, le=9),
            num_tokens: int | None = Form(None, ge=1200, le=3600),
            refine_steps: int = Form(3, ge=0, le=8),
            fov_x: float | None = Form(None, gt=0.0, lt=180.0),
            edge_threshold: float = Form(0.04, ge=0.0),
        ):
            started_at = time.perf_counter()
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds 25 MiB")

            image_array = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            original_height, original_width = image.shape[:2]
            if max(original_height, original_width) > max_size:
                scale = max_size / max(original_height, original_width)
                resized_width = max(1, round(original_width * scale))
                resized_height = max(1, round(original_height * scale))
                image = cv2.resize(
                    image,
                    (resized_width, resized_height),
                    interpolation=cv2.INTER_AREA,
                )

            height, width = image.shape[:2]
            image_tensor = (
                torch.from_numpy(image.copy())
                .to(device="cuda", dtype=torch.float32)
                .permute(2, 0, 1)
                / 255.0
            )
            output = self.model.infer(
                image_tensor,
                apply_mask=True,
                fov_x=fov_x,
                num_tokens=num_tokens,
                refine_steps=refine_steps,
                resolution_level=resolution_level,
                use_fp16=True,
            )

            points = output["points"].cpu().numpy()
            depth = output["depth"].cpu().numpy()
            mask = output["mask"].cpu().numpy()
            intrinsics = output["intrinsics"].cpu().numpy()
            normal_tensor = output.get("normal")
            normal = normal_tensor.cpu().numpy() if normal_tensor is not None else None

            mask_cleaned = mask & ~utils3d.np.depth_map_edge(
                depth,
                rtol=edge_threshold,
            )

            coordinate_flip = np.array([1, -1, -1], dtype=np.float32)
            point_map = points.astype(np.float32, copy=True) * coordinate_flip
            point_map[~mask_cleaned] = np.nan
            point_map_buffer = io.BytesIO()
            np.save(point_map_buffer, point_map, allow_pickle=False)

            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(
                archive_buffer,
                mode="w",
                compression=zipfile.ZIP_STORED,
            ) as archive:
                if include_mesh:
                    archive.writestr(
                        "output.glb",
                        _mesh_glb(
                            image,
                            points,
                            normal,
                            mask_cleaned,
                            intrinsics,
                        ),
                    )

                archive.writestr("point_map.npy", point_map_buffer.getvalue())

                if include_debug:
                    depth_visualization = depth.copy()
                    depth_visualization[~mask_cleaned] = np.nan
                    archive.writestr(
                        "depth.png",
                        _png_bytes(colorize_depth(depth_visualization)),
                    )
                    if normal is not None:
                        normal_visualization = normal.copy()
                        normal_visualization[~mask_cleaned] = 0
                        archive.writestr(
                            "normal.png",
                            _png_bytes(colorize_normal(normal_visualization)),
                        )

                fov_x_deg = float(
                    np.rad2deg(2.0 * np.arctan(0.5 / intrinsics[0, 0]))
                )
                fov_y_deg = float(
                    np.rad2deg(2.0 * np.arctan(0.5 / intrinsics[1, 1]))
                )
                metadata = {
                    "model": MODEL_REPO,
                    "source_revision": SOURCE_REVISION,
                    "original_image_size": [original_width, original_height],
                    "image_size": [width, height],
                    "intrinsics": intrinsics.tolist(),
                    "intrinsics_convention": (
                        "normalized by image width/height "
                        "(principal_point=0.5,0.5)"
                    ),
                    "fov_x_deg": fov_x_deg,
                    "fov_y_deg": fov_y_deg,
                    "point_map_coordinates": (
                        "output.glb camera coordinates "
                        "(+X right, +Y up, -Z forward)"
                    ),
                    "inference": {
                        "edge_threshold": edge_threshold,
                        "max_size": max_size,
                        "num_tokens": num_tokens,
                        "refine_steps": refine_steps,
                        "resolution_level": resolution_level,
                    },
                }
                archive.writestr(
                    "metadata.json",
                    json.dumps(metadata, indent=2),
                )

            archive_buffer.seek(0)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            print(
                f"[{file.filename}] {width}x{height}, "
                f"refine_steps={refine_steps} -> {elapsed_ms:.0f} ms"
            )
            return StreamingResponse(
                archive_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": "attachment; filename=output.zip",
                    "Server-Timing": f"moge;dur={elapsed_ms:.1f}",
                },
            )

        @web_app.get("/health")
        async def health():
            return {
                "model": MODEL_REPO,
                "source_revision": SOURCE_REVISION,
                "status": "ok",
            }

        return web_app
