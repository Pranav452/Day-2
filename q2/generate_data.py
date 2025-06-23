import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random

def generate_data():
    """Generate 5 prompts with 4 candidate answers each, ranked 1-4."""
    
    # 5 prompts as required
    prompts = [
        "Tell me a funny joke about programming:",
        "Write a short summary of the benefits of renewable energy:",
        "Explain why reading books is important in 2-3 sentences:",
        "Create a mini-essay about the importance of teamwork (50-100 words):",
        "Describe your ideal vacation destination and why:"
    ]
    
    # Realistic answers for each prompt with different quality levels
    answers_by_prompt = {
        "Tell me a funny joke about programming:": [
            "Why do programmers prefer dark mode? Because light attracts bugs! It's also easier on the eyes during those long coding sessions, and honestly, we just look cooler typing in the darkness.",  # Rank 4
            "Why don't programmers like nature? It has too many bugs! Also, there's no Wi-Fi in the woods.",  # Rank 3
            "Programming is like a joke. If you have to explain it, it's not that good.",  # Rank 2
            "Code funny sometimes."  # Rank 1
        ],
        
        "Write a short summary of the benefits of renewable energy:": [
            "Renewable energy sources like solar, wind, and hydroelectric power offer significant environmental and economic advantages. They reduce greenhouse gas emissions, decrease dependence on fossil fuels, create sustainable jobs, and provide long-term cost savings. Additionally, renewable energy enhances energy security and helps combat climate change while promoting technological innovation.",  # Rank 4
            "Renewable energy is good for the environment because it doesn't produce harmful emissions. It includes solar panels, wind turbines, and water power. It's also becoming cheaper and creates new jobs.",  # Rank 3
            "Renewable energy is clean and doesn't pollute the air like coal or oil. It's better for nature.",  # Rank 2
            "Clean energy good."  # Rank 1
        ],
        
        "Explain why reading books is important in 2-3 sentences:": [
            "Reading books enhances cognitive function, improves vocabulary, and broadens knowledge across diverse subjects while developing critical thinking skills. It also fosters empathy by exposing readers to different perspectives and experiences, ultimately making us more well-rounded individuals. Regular reading has been shown to reduce stress and improve mental health throughout life.",  # Rank 4
            "Reading books helps you learn new things and makes you smarter. It also improves your vocabulary and helps you understand different viewpoints from various authors and characters.",  # Rank 3
            "Books are good because they teach you stuff and make you think better.",  # Rank 2
            "Reading important."  # Rank 1
        ],
        
        "Create a mini-essay about the importance of teamwork (50-100 words):": [
            "Teamwork is fundamental to achieving complex goals that exceed individual capabilities. When people collaborate effectively, they combine diverse skills, perspectives, and experiences to solve problems more creatively and efficiently. Good teamwork fosters communication, builds trust, and creates shared accountability. It distributes workload, reduces individual stress, and often leads to superior outcomes than solo efforts. In professional settings, teamwork drives innovation and productivity, while in personal contexts, it strengthens relationships and builds community bonds that enhance everyone's success.",  # Rank 4
            "Teamwork is important because it helps people work together to achieve common goals. When team members combine their different skills and ideas, they can solve problems better than working alone. Good teamwork requires communication, trust, and cooperation between all members to be successful.",  # Rank 3
            "Teams are good because people can help each other. Working together is better than working alone because you get more done.",  # Rank 2
            "Teamwork good."  # Rank 1
        ],
        
        "Describe your ideal vacation destination and why:": [
            "My ideal vacation destination would be New Zealand, combining breathtaking natural landscapes with adventure activities and rich cultural experiences. The country offers diverse experiences from snow-capped mountains and pristine lakes to geothermal wonders and stunning coastlines. I'm drawn to its excellent hiking trails, friendly locals, unique wildlife, and the perfect balance of relaxation and adventure sports like bungee jumping and skydiving. The clean environment and sustainable tourism practices make it an ideal escape from city life.",  # Rank 4
            "I would love to visit Japan because it has beautiful temples, delicious food, and fascinating culture. The cherry blossoms in spring look amazing, and I'm interested in both traditional and modern aspects of Japanese society.",  # Rank 3
            "A beach vacation sounds nice because it's relaxing and you can swim and get a tan.",  # Rank 2
            "Somewhere fun."  # Rank 1
        ]
    }
    
    data = []
    
    for prompt in prompts:
        print(f"Processing: {prompt[:40]}...")
        
        # Get the specific answers for this prompt
        candidates = answers_by_prompt[prompt]
        
        # Add to dataset with ranks (4=best, 1=worst)
        for rank, answer in enumerate(candidates, 1):
            data.append({
                'prompt': prompt,
                'answer': answer,
                'rank': 5 - rank  # Convert to 4=best, 1=worst
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('answers.csv', index=False)
    
    print(f"Generated {len(df)} entries saved to answers.csv")
    print(f"Rank distribution:\n{df['rank'].value_counts().sort_index()}")
    
    return df

if __name__ == "__main__":
    generate_data() 