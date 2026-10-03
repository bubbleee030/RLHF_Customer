import json
import os
from pathlib import Path

def parse_conversation(log_file, output_file, title):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# {title}\n\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            source = data.get("source")
            content = data.get("content")
            
            if content:
                if source == "USER_EXPLICIT" or source == "USER":
                    # Clean up <USER_REQUEST> tags if they exist to make it more readable
                    display_content = content
                    if "<USER_REQUEST>" in content:
                        start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
                        end = content.find("</USER_REQUEST>")
                        if end != -1:
                            display_content = content[start:end].strip()
                    
                    out.write(f"## 🧑 User\n\n{display_content}\n\n---\n\n")
                elif source == "MODEL":
                    out.write(f"## 🤖 Model\n\n{content}\n\n---\n\n")

conv1 = "/home/ubuntu/.gemini/antigravity/brain/146b9fca-502d-4ddc-9a9e-55e9a4043322/.system_generated/logs/overview.txt"
conv2 = "/home/ubuntu/.gemini/antigravity/brain/59b2d6f0-857a-4dc8-a62f-49e6e2a0413b/.system_generated/logs/overview.txt"

parse_conversation(conv1, "/home/ubuntu/reward_model/conversation_1_Cost_Model_Training.md", "Conversation 1: Training Cost Model for Safety Alignment")
parse_conversation(conv2, "/home/ubuntu/reward_model/conversation_2_HPC_Deployment.md", "Conversation 2: HPC Training Pipeline Deployment")
print("Conversations successfully rebuilt as markdown files.")
