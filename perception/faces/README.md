# Face Recognition

Face recognition uses **embeddings** instead of reference images. No manual setup required.

## How It Works

1. **First time a face is seen** → Embedding is computed and stored → Assigned `user_1`
2. **Same face seen again** → Embedding matched against stored → Returns `user_1`
3. **New person** → No match → New embedding stored → Assigned `user_2`, `user_3`, ...

Embeddings are stored in `face_embeddings.json` (project root). No face images are saved.

---

## Enrolling a Person from a Folder of Images

Use `perception/enroll_from_folder.py` to bulk-register a person from existing photos.
This is useful when the person is not physically present, or to pre-load multiple angles
and lighting conditions for better recognition accuracy.

```bash
python perception/enroll_from_folder.py --name "Alice" --folder /path/to/alice_photos/
```

### What it does

1. Scans the folder for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`)
2. Extracts a Facenet512 embedding from each image via DeepFace
3. If a photo contains multiple faces, uses the largest (and warns you)
4. Adds all extracted embeddings to `face_embeddings.json` under the given name
5. Auto-merges any anonymous `user_N` entries whose embeddings are similar enough (≥ 0.62 cosine similarity)
6. Prints a summary of what was added, skipped, and merged

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | *(required)* | Name to enroll (e.g. `"Alice"`) |
| `--folder` | *(required)* | Directory of face images |
| `--embeddings-path` | `face_embeddings.json` | Path to the embeddings file |
| `--detector` | `yunet` | DeepFace detector backend (`yunet`, `opencv`, `retinaface`, `mtcnn`, …) |
| `--model` | `Facenet512` | DeepFace embedding model |
| `--max-per-user` | `20` | Max embeddings stored per person |

Defaults for `--detector`, `--model`, and `--max-per-user` can also be set via environment variables:
`DEEPFACE_DETECTOR_BACKEND`, `EMBEDDING_MODEL`, `MAX_EMBEDDINGS_PER_USER`.

### Example output

```
Found 5 image(s) in /path/to/alice_photos. Extracting embeddings…

  [OK]   alice_front.jpg
  [OK]   alice_side.jpg
  [SKIP] no_face.jpg          ← no face detected
  [OK]   alice_glasses.jpg
WARNING: Multiple faces detected — using the largest.
  [OK]   alice_group.jpg

Extracted 4 embedding(s), skipped 1 image(s).

✓ Enrolled 'Alice'
  Embeddings added this run : 4
  Total embeddings for 'Alice': 4
  Merged anonymous entries  : user_3
  Saved to: /path/to/face_embeddings.json
```

### Tips

- **More photos = better accuracy.** Aim for varied angles, lighting, and expressions.
- Up to 20 embeddings are stored per person (configurable with `--max-per-user`). Beyond that, new embeddings are silently dropped.
- If the person already exists in the database, new embeddings are appended — existing ones are never overwritten.
- If you see `[SKIP]` for every image, try `--detector opencv` as a fallback (slower but more permissive than `yunet`).

---

## Enrolling via the Live Camera

Send a POST request to the perception service while the person is in frame:

```bash
curl -X POST http://localhost:8089/register-face \
  -F "name=Alice" \
  -F "file=@photo.jpg"
```

Or omit the file to use the current camera frame:

```bash
curl -X POST http://localhost:8089/register-face \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

---

## Runtime Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `EMBEDDINGS_PATH` | `face_embeddings.json` | Where embeddings are stored |
| `SIMILARITY_THRESHOLD` | `0.65` | Min similarity to match a known face (0.5–0.7 typical) |
| `MERGE_SIMILARITY_THRESHOLD` | `0.62` | Min similarity to merge `user_N` into a named person |
| `MAX_EMBEDDINGS_PER_USER` | `20` | Max embeddings stored per person |
| `EMBEDDING_MODEL` | `Facenet512` | DeepFace model used for all embedding extraction |
| `DEEPFACE_DETECTOR_BACKEND` | `yunet` | Face detector used during live recognition |

---

## Storage Format (`face_embeddings.json`)

```json
{
  "Alice": {
    "embeddings": [
      [0.638, 1.192, ...],
      [0.540, 1.205, ...]
    ],
    "first_seen": 1775771988.25
  },
  "user_1": {
    "embeddings": [[...]],
    "first_seen": 1775700000.0
  }
}
```

- Each embedding is a 512-dimensional float vector (Facenet512)
- Named entries (e.g. `"Alice"`) are registered people
- Anonymous entries (`user_N`) are faces seen by the camera that haven't been named yet

---

## Legacy: Reference Images (`faces/` folder)

The `faces/` folder is no longer used. Face recognition is fully embedding-based.
