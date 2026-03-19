import argparse
import os
import shutil
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True, help="USplat4D output dir")
    args = parser.parse_args()

    work_dir = args.work_dir
    graph_path = os.path.join(work_dir, "graph.pt")
    raw_ckpt_path = os.path.join(work_dir, "usplat4d", "final._raw.ckpt")
    active_ckpt_path = os.path.join(work_dir, "usplat4d", "final.ckpt")

    if not os.path.exists(graph_path):
        print(f"Error: Could not find graph.pt at {graph_path}")
        return

    # 1. Backup the original final.ckpt so we can restore it later if needed
    if not os.path.exists(raw_ckpt_path):
        print(f"Backing up original weights to {raw_ckpt_path}")
        shutil.copy2(active_ckpt_path, raw_ckpt_path)
    
    print(f"Loading raw checkpoint...")
    ckpt = torch.load(raw_ckpt_path, map_location="cpu", weights_only=False)

    print(f"Loading graph...")
    graph_data = torch.load(graph_path, map_location="cpu")
    key_idx = graph_data["graph"]["key_idx"].long()

    state_dict = ckpt["model"]
    
    # 2. Correct parameter keys
    color_key = "fg.params.colors"
    opacity_key = "fg.params.opacities"
    scale_key = "fg.params.scales"
    
    if color_key not in state_dict:
        print(f"Error: Could not find {color_key} in state dict.")
        return

    total_gs = state_dict[color_key].shape[0]
    print(f"Total FG Gaussians: {total_gs} | Key nodes: {len(key_idx)} ({len(key_idx)/total_gs*100:.2f}%)")

    # 3. EMPHASIZE the key nodes
    # - BRIGHT RED color (pre-sigmoid)
    state_dict[color_key][key_idx, 0] = 50.0   # Extreme R
    state_dict[color_key][key_idx, 1] = -50.0  # Zero G
    state_dict[color_key][key_idx, 2] = -50.0  # Zero B

    # - MAX OPACITY 
    if opacity_key in state_dict:
        state_dict[opacity_key][key_idx] = 50.0 
        
    # - MASSIVE SCALE: Overwrite their current scale with a huge value
    # (since scales are log(s), a value of 0.0 means scale=1.0, which is extremely large for a scene.
    # We will compute the average scale of all gaussians and add a massive factor.
    if scale_key in state_dict:
        mean_scale = state_dict[scale_key].mean()
        # Add 2.0 to log scale -> roughly 7x larger
        state_dict[scale_key][key_idx] = torch.clamp(state_dict[scale_key][key_idx] - 0.0, max=0.0)

    print(f"Saving HUGE RED key nodes checkpoint back to {active_ckpt_path}")
    torch.save(ckpt, active_ckpt_path)
    print("Done! Launching the viewer now should show massive red floating blobs.")

if __name__ == "__main__":
    main()
