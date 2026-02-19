import ollama
from googleapiclient.discovery import build

# --- Configuration ---
class Config:
    GOOGLE_API_KEY = "56703e4e06c64fc79a70991634d5b135.mz0CCPsP7ZuWjw18g621frwG"
    SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"
    # The 120B model is a hybrid-cloud model in Ollama 2026
    MODEL = "gpt-oss:120b-cloud" 

def search_medicine_info(medicine_name: str):
    """Searches Google for medicine details, side effects, and alternatives."""
    try:
        service = build("customsearch", "v1", developerKey=Config.GOOGLE_API_KEY)
        query = f"{medicine_name} uses side effects medical"
        res = service.cse().list(q=query, cx=Config.SEARCH_ENGINE_ID, num=3).execute()
        
        items = res.get('items', [])
        if not items: return "No online information found."
        
        return "\n\n".join([f"Source: {i['title']}\n{i['snippet']}" for i in items])
    except Exception as e:
        return f"Search Error: {str(e)}"

def main():
    print(f"--- Medical Agent Ready (Ollama: {Config.MODEL}) ---")
    
    # Store conversation history
    messages = [
        {'role': 'system', 'content': 'You are a medical assistant. Use the search tool for medicine info.'}
    ]

    while True:
        user_input = input("\nMedicine Name: ").strip()
        if user_input.lower() in ['exit', 'quit']: break

        messages.append({'role': 'user', 'content': user_input})

        try:
            # 1. Ask the model
            # In Ollama 0.4+, passing functions directly is supported
            response = ollama.chat(
                model=Config.MODEL,
                messages=messages,
                tools=[search_medicine_info] 
            )

            # 2. Check for Tool Calls
            if response.message.tool_calls:
                for call in response.message.tool_calls:
                    print(f"[*] Searching for info on {user_input}...")
                    
                    # Execute tool
                    tool_output = search_medicine_info(call.function.arguments['medicine_name'])
                    
                    # Add model's request and tool's response to history
                    messages.append(response.message)
                    messages.append({'role': 'tool', 'content': tool_output})

                # 3. Get final summarized answer
                final_response = ollama.chat(model=Config.MODEL, messages=messages)
                print(f"\nAssistant: {final_response.message.content}")
                messages.append(final_response.message)
            else:
                print(f"\nAssistant: {response.message.content}")
                messages.append(response.message)

        except Exception as e:
            print(f"System Error: {e}")

if __name__ == "__main__":
    main()