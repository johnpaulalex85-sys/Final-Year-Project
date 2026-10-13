import logging
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class QwenService:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        logger.info(f"Loading Qwen model {self.model_name}...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use bfloat16 for GPU to save VRAM and speed up inference
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                torch_dtype=torch_dtype, 
                device_map="auto" if device == "cuda" else None
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("Qwen model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.model = None
            self.processor = None

    def generate_description(self, frame: np.ndarray, scene_json: dict) -> str:
        """
        Generates a description using Qwen2-VL based on the frame and scene data.
        frame should be a BGR numpy array (OpenCV format) or RGB PIL Image.
        """
        if self.model is None or self.processor is None:
            return "Vision Language Model is not initialized."
            
        try:
            # If the input is a numpy array from OpenCV (BGR), convert to PIL RGB
            if isinstance(frame, np.ndarray):
                import cv2
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
            else:
                pil_image = frame

            prompt = (
                "You are assisting a visually impaired person.\n"
                "Describe the surroundings clearly.\n\n"
                "Mention:\n"
                "- nearby people\n"
                "- obstacles\n"
                "- important objects\n"
                "- readable text\n"
                "- approximate distances\n\n"
                "Keep the response under 80 words.\n\n"
                f"Scene Data Context:\n{json.dumps(scene_json, indent=2)}"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)

            # Generate (max_new_tokens is ~80 words + buffer)
            generated_ids = self.model.generate(**inputs, max_new_tokens=120)
            
            # Trim the generated_ids to only the new output tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            logger.error(f"Qwen generation error: {e}")
            return "An error occurred while generating the description."
