from pathlib import Path

pairs = [
    (0, 1),   # wrists
    (2, 3),   # elbows
    (4, 5),   # shoulders
    (9, 10),  # hips
    (12, 13), # knees
    (14, 15), # ankles
    (16, 17)  # foot index
]

for file in Path(".").glob("*.txt"):
    new_lines = []

    for line in open(file, encoding="utf-8").read().strip().splitlines():
        parts = line.strip().split()

        # Need at least bbox + some keypoints
        if len(parts) <= 5:
            new_lines.append(line)
            continue

        header = parts[:5]
        kps = parts[5:]

        # Split into x,y,v groups
        kp_triplets = [kps[i:i+3] for i in range(0, len(kps), 3)]

        # Swap only if both indexes exist
        for a, b in pairs:
            if a < len(kp_triplets) and b < len(kp_triplets):
                kp_triplets[a], kp_triplets[b] = kp_triplets[b], kp_triplets[a]

        flattened = [x for triplet in kp_triplets for x in triplet]

        new_lines.append(" ".join(header + flattened))

    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

print("Done")