import json
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
import google.auth

import openai
import requests
import json
import os
import google.auth
import google.auth.transport.requests
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage



with open("config.json", "r") as f:
    CONFIG = json.load(f)

def get_llm(model: str, api_key: str = None, project_id: str = None, location: str = None):
    if model.startswith("gpt") or model == "o4-mini":
        return ChatOpenAI(model_name=model, openai_api_key=api_key or CONFIG["openai_api_key"])
    elif model.startswith("vertex-"):
        model_id = model.replace("vertex-", "", 1).strip()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return ChatVertexAI(
            model_name=model_id,
            project=project_id,
            location=location,
            credentials=credentials
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


def llm_call(llm, messages, use_logprobs: bool = False, log_dir: str = None) -> str:
    if use_logprobs and isinstance(llm, ChatOpenAI):
        try:
            result = llm.generate([messages], logprobs=True, top_logprobs=5)
            generation = result.generations[0][0]
            content = generation.message.content
            logprobs = generation.generation_info.get('logprobs', None)
            if logprobs and log_dir:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                logprob_file = os.path.join(log_dir, f"logprobs_{timestamp_str}.json")
                formatted_logprobs = {
                    "full_text": content,
                    "tokens": {
                        "content": [
                            {
                                "token": token["token"],
                                "bytes": token["bytes"],
                                "logprob": token["logprob"],
                                "top_logprobs": token.get("top_logprobs", [])
                            }
                            for token in logprobs["content"]
                        ]
                    }
                }
                with open(logprob_file, "w") as f:
                    json.dump(formatted_logprobs, f, indent=2)
            elif not logprobs and log_dir:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                debug_file = os.path.join(log_dir, f"logprobs_debug_{timestamp_str}.txt")
                with open(debug_file, "w") as f:
                    f.write(f"No logprobs returned by LLM. Response structure: {generation.__dict__}")
        except Exception as e:
            if log_dir:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                debug_file = os.path.join(log_dir, f"logprobs_debug_{timestamp_str}.txt")
                with open(debug_file, "w") as f:
                    f.write(f"Error requesting logprobs: {str(e)}")
            raise
    else:
        response = llm.invoke(messages)
        content = response.content
    return content