---
sidebar_position: 1
---

# Voice-to-Action

## Overview

Voice-to-action systems enable robots to understand natural language commands and translate them into executable robotic actions. This is a critical component of Vision-Language-Action (VLA) systems, allowing for intuitive human-robot interaction through speech.

## System Architecture

A complete voice-to-action system consists of several components:

```
Speech Input → Speech Recognition → Natural Language Processing → Action Mapping → Robot Execution
```

### Key Components
- **Speech Recognition**: Converting speech to text
- **Natural Language Understanding**: Interpreting user intent
- **Action Planning**: Mapping language to robot actions
- **Execution**: Carrying out the planned actions

## Speech Recognition

### Overview
Speech recognition converts spoken language into text. Modern systems use deep learning models trained on large datasets.

### Using OpenAI Whisper
OpenAI Whisper is a state-of-the-art speech recognition model:

```python
import openai
import speech_recognition as sr
import pyaudio
import wave
import os

class SpeechRecognizer:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def listen_and_transcribe(self):
        """Listen to audio and transcribe using Whisper"""
        print("Listening...")
        with self.microphone as source:
            audio = self.recognizer.listen(source, timeout=5)

        # Save audio to temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        # Use OpenAI Whisper API
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )

        # Clean up temporary file
        os.remove("temp_audio.wav")

        return transcript.text

    def recognize_with_local_model(self):
        """Alternative: Use local speech recognition"""
        with self.microphone as source:
            audio = self.recognizer.listen(source, timeout=5)

        try:
            # Use Google's speech recognition (requires internet)
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"
```

### Alternative: Using Hugging Face Transformers
```python
from transformers import pipeline
import librosa
import numpy as np

class LocalSpeechRecognizer:
    def __init__(self):
        # Load pre-trained speech recognition pipeline
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny"
        )

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using local model"""
        result = self.asr_pipeline(audio_path)
        return result["text"]

    def transcribe_array(self, audio_array, sampling_rate=16000):
        """Transcribe audio array"""
        result = self.asr_pipeline({
            "array": audio_array,
            "sampling_rate": sampling_rate
        })
        return result["text"]
```

## Natural Language Understanding (NLU)

### Intent Recognition
Identifying the user's intent from the transcribed text:

```python
import spacy
import re
from typing import Dict, List, Tuple

class IntentRecognizer:
    def __init__(self):
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define intent patterns
        self.intent_patterns = {
            'move': [
                r'move\s+(?P<direction>\w+)\s*(?P<distance>[\d.]+)?\s*(?P<unit>\w+)?',
                r'go\s+(?P<direction>\w+)\s*(?P<distance>[\d.]+)?\s*(?P<unit>\w+)?',
                r'walk\s+(?P<direction>\w+)\s*(?P<distance>[\d.]+)?\s*(?P<unit>\w+)?'
            ],
            'grasp': [
                r'pick\s+up\s+(?P<object>[\w\s]+)',
                r'grab\s+(?P<object>[\w\s]+)',
                r'lift\s+(?P<object>[\w\s]+)',
                r'hold\s+(?P<object>[\w\s]+)'
            ],
            'place': [
                r'put\s+(?P<object>[\w\s]+)\s+on\s+(?P<location>[\w\s]+)',
                r'place\s+(?P<object>[\w\s]+)\s+on\s+(?P<location>[\w\s]+)',
                r'set\s+(?P<object>[\w\s]+)\s+on\s+(?P<location>[\w\s]+)'
            ],
            'navigate': [
                r'go\s+to\s+(?P<location>[\w\s]+)',
                r'go\s+to\s+the\s+(?P<location>[\w\s]+)',
                r'navigate\s+to\s+(?P<location>[\w\s]+)',
                r'move\s+to\s+(?P<location>[\w\s]+)'
            ],
            'inspect': [
                r'look\s+at\s+(?P<object>[\w\s]+)',
                r'check\s+(?P<object>[\w\s]+)',
                r'examine\s+(?P<object>[\w\s]+)',
                r'observe\s+(?P<object>[\w\s]+)'
            ]
        }

    def extract_intent(self, text: str) -> Dict:
        """Extract intent and parameters from text"""
        text = text.lower().strip()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    params = match.groupdict()
                    return {
                        'intent': intent,
                        'parameters': params,
                        'confidence': 0.9  # Simplified confidence
                    }

        # If no pattern matches, use NLP for more complex understanding
        return self.nlp_intent_extraction(text)

    def nlp_intent_extraction(self, text: str) -> Dict:
        """Use NLP for more complex intent extraction"""
        if not self.nlp:
            return {'intent': 'unknown', 'parameters': {}, 'confidence': 0.0}

        doc = self.nlp(text)

        # Extract named entities and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        tokens = [(token.text, token.pos_, token.dep_) for token in doc]

        # Simple heuristic for intent detection
        verbs = [token.text for token in doc if token.pos_ == 'VERB']
        objects = [token.text for token in doc if token.pos_ == 'NOUN']

        intent = 'unknown'
        if any(v in ['move', 'go', 'walk', 'navigate'] for v in verbs):
            intent = 'navigate'
        elif any(v in ['pick', 'grab', 'lift', 'hold'] for v in verbs):
            intent = 'grasp'
        elif any(v in ['put', 'place', 'set'] for v in verbs):
            intent = 'place'

        return {
            'intent': intent,
            'parameters': {
                'entities': entities,
                'verbs': verbs,
                'objects': objects
            },
            'confidence': 0.7
        }
```

## Action Planning and Mapping

### Converting Language to Actions
Mapping natural language commands to specific robot actions:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Union, List

class RobotActionType(Enum):
    MOVE_BASE = "move_base"
    ARM_MANIPULATION = "arm_manipulation"
    GRIPPER_CONTROL = "gripper_control"
    HEAD_CONTROL = "head_control"
    SPEAK = "speak"

@dataclass
class RobotAction:
    action_type: RobotActionType
    parameters: dict
    priority: int = 1

class ActionMapper:
    def __init__(self):
        self.location_map = {
            'kitchen': [1.0, 2.0, 0.0],
            'living room': [3.0, 1.0, 0.0],
            'bedroom': [5.0, 3.0, 0.0],
            'table': [2.5, 2.5, 0.0],
            'shelf': [4.0, 1.5, 0.0]
        }

        self.object_map = {
            'cup': 'cup_01',
            'book': 'book_01',
            'bottle': 'bottle_01',
            'box': 'box_01'
        }

    def map_intent_to_action(self, intent_result: Dict) -> List[RobotAction]:
        """Map intent to specific robot actions"""
        intent = intent_result['intent']
        params = intent_result['parameters']

        actions = []

        if intent == 'navigate':
            location = params.get('location', '').strip()
            if location in self.location_map:
                target_pose = self.location_map[location]
                actions.append(RobotAction(
                    action_type=RobotActionType.MOVE_BASE,
                    parameters={
                        'target_pose': target_pose,
                        'frame_id': 'map'
                    }
                ))
            else:
                # Try to find closest match
                target_pose = self.find_closest_location(location)
                if target_pose:
                    actions.append(RobotAction(
                        action_type=RobotActionType.MOVE_BASE,
                        parameters={
                            'target_pose': target_pose,
                            'frame_id': 'map'
                        }
                    ))

        elif intent == 'grasp':
            object_name = params.get('object', '').strip()
            if object_name in self.object_map:
                object_id = self.object_map[object_name]
                actions.extend([
                    RobotAction(
                        action_type=RobotActionType.MOVE_BASE,
                        parameters={'target_object': object_id}
                    ),
                    RobotAction(
                        action_type=RobotActionType.ARM_MANIPULATION,
                        parameters={'action': 'pick', 'object_id': object_id}
                    ),
                    RobotAction(
                        action_type=RobotActionType.GRIPPER_CONTROL,
                        parameters={'action': 'close'}
                    )
                ])

        elif intent == 'place':
            object_name = params.get('object', '').strip()
            location = params.get('location', '').strip()

            if location in self.location_map:
                target_pose = self.location_map[location]
                actions.extend([
                    RobotAction(
                        action_type=RobotActionType.MOVE_BASE,
                        parameters={'target_pose': target_pose}
                    ),
                    RobotAction(
                        action_type=RobotActionType.ARM_MANIPULATION,
                        parameters={'action': 'place', 'target_pose': target_pose}
                    ),
                    RobotAction(
                        action_type=RobotActionType.GRIPPER_CONTROL,
                        parameters={'action': 'open'}
                    )
                ])

        elif intent == 'move':
            direction = params.get('direction', 'forward')
            distance = float(params.get('distance', 1.0))
            unit = params.get('unit', 'meters')

            # Convert to robot-appropriate units if needed
            if unit.lower() in ['m', 'meters']:
                distance_m = distance
            elif unit.lower() in ['cm', 'centimeters']:
                distance_m = distance / 100.0
            else:
                distance_m = distance  # Default to meters

            move_vector = self.direction_to_vector(direction, distance_m)
            actions.append(RobotAction(
                action_type=RobotActionType.MOVE_BASE,
                parameters={
                    'move_vector': move_vector
                }
            ))

        return actions

    def direction_to_vector(self, direction: str, distance: float) -> List[float]:
        """Convert direction string to movement vector"""
        direction = direction.lower()
        if direction in ['forward', 'front', 'ahead']:
            return [distance, 0, 0]
        elif direction in ['backward', 'back']:
            return [-distance, 0, 0]
        elif direction in ['left', 'port']:
            return [0, distance, 0]
        elif direction in ['right', 'starboard']:
            return [0, -distance, 0]
        elif direction in ['up', 'raise']:
            return [0, 0, distance]
        elif direction in ['down', 'lower']:
            return [0, 0, -distance]
        else:
            # Default to forward if unknown direction
            return [distance, 0, 0]

    def find_closest_location(self, location_name: str) -> Union[List[float], None]:
        """Find closest location to the given name"""
        # Simplified implementation - in practice, you'd use more sophisticated matching
        for loc_name, coords in self.location_map.items():
            if location_name.lower() in loc_name.lower() or loc_name.lower() in location_name.lower():
                return coords
        return None
```

## Integration with ROS 2

### Voice Command Node
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)

        # Initialize components
        self.speech_recognizer = SpeechRecognizer()
        self.intent_recognizer = IntentRecognizer()
        self.action_mapper = ActionMapper()

        # Timer for continuous listening
        self.timer = self.create_timer(1.0, self.listen_callback)

    def listen_callback(self):
        """Continuously listen for voice commands"""
        try:
            # Listen for command
            command_text = self.speech_recognizer.listen_and_transcribe()
            self.get_logger().info(f"Heard: {command_text}")

            # Process command
            intent_result = self.intent_recognizer.extract_intent(command_text)
            actions = self.action_mapper.map_intent_to_action(intent_result)

            # Execute actions
            for action in actions:
                self.execute_action(action)

        except Exception as e:
            self.get_logger().error(f"Error processing voice command: {e}")

    def execute_action(self, action: RobotAction):
        """Execute a specific robot action"""
        if action.action_type == RobotActionType.MOVE_BASE:
            self.execute_move_base_action(action.parameters)
        elif action.action_type == RobotActionType.SPEAK:
            self.speak(action.parameters.get('text', ''))
        # Add other action types as needed

    def execute_move_base_action(self, params: dict):
        """Execute base movement action"""
        twist_msg = Twist()

        if 'target_pose' in params:
            # For navigation to specific pose, you'd typically use navigation2
            self.navigate_to_pose(params['target_pose'])
        elif 'move_vector' in params:
            # Simple relative movement
            vector = params['move_vector']
            twist_msg.linear.x = vector[0]
            twist_msg.linear.y = vector[1]
            twist_msg.linear.z = vector[2]
            self.cmd_vel_pub.publish(twist_msg)

    def navigate_to_pose(self, target_pose):
        """Navigate to a specific pose using Nav2"""
        # This would typically use the NavigateToPose action
        # Implementation depends on your navigation setup
        pass

    def speak(self, text: str):
        """Publish text for TTS"""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    voice_node = VoiceCommandNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()
```

## Advanced NLP with LLMs

### Using OpenAI GPT for Complex Understanding
For more sophisticated language understanding:

```python
import openai
import json

class AdvancedLanguageProcessor:
    def __init__(self, api_key):
        openai.api_key = api_key

    def process_command(self, command: str, robot_capabilities: dict) -> dict:
        """Use LLM to process complex commands"""
        prompt = f"""
        You are a robot command interpreter. Convert the following human command into structured robot actions.

        Robot capabilities: {json.dumps(robot_capabilities, indent=2)}

        Human command: "{command}"

        Respond with a JSON object containing:
        1. intent: The main intent of the command
        2. parameters: Relevant parameters for the action
        3. sequence: Sequence of actions to execute
        4. validation: How to verify the command was executed successfully

        Respond with only valid JSON:
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        try:
            result = json.loads(response.choices[0].message.content.strip())
            return result
        except json.JSONDecodeError:
            # Fallback to simple processing
            return self.fallback_process(command)

    def fallback_process(self, command: str) -> dict:
        """Fallback processing if LLM fails"""
        return {
            "intent": "unknown",
            "parameters": {"raw_command": command},
            "sequence": [],
            "validation": "unknown"
        }

# Example usage
robot_caps = {
    "navigation": ["move_to_location", "move_direction"],
    "manipulation": ["pick_object", "place_object", "gripper_control"],
    "sensors": ["camera", "lidar", "microphone"],
    "communication": ["speak", "listen"]
}

processor = AdvancedLanguageProcessor("your-api-key")
result = processor.process_command("Please go to the kitchen and bring me a cup from the table", robot_caps)
```

## Voice Activity Detection

### Detecting When User is Speaking
```python
import pyaudio
import numpy as np
from scipy import signal

class VoiceActivityDetector:
    def __init__(self, threshold=0.01, silence_duration=1.0):
        self.threshold = threshold  # Energy threshold for speech detection
        self.silence_duration = silence_duration  # Duration of silence to stop
        self.audio = pyaudio.PyAudio()

    def detect_voice_activity(self, duration=5.0):
        """Detect voice activity and return audio data"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        frames = []
        silence_frames = 0
        max_silence_frames = int(self.silence_duration * 16000 / 1024)

        print("Listening for voice activity...")

        while len(frames) < duration * 16000 / 1024:
            data = stream.read(1024)
            frames.append(data)

            # Convert to numpy array for analysis
            audio_data = np.frombuffer(data, dtype=np.int16)
            energy = np.sum(audio_data ** 2) / len(audio_data)

            if energy < self.threshold:
                silence_frames += 1
                if silence_frames > max_silence_frames:
                    print("Silence detected, stopping...")
                    break
            else:
                silence_frames = 0  # Reset silence counter when speech detected

        stream.stop_stream()
        stream.close()

        return b''.join(frames)

    def __del__(self):
        self.audio.terminate()
```

## Error Handling and Robustness

### Handling Uncertainty and Errors
```python
class RobustVoiceProcessor:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.max_retries = 3

    def process_with_confirmation(self, user_command: str):
        """Process command with user confirmation for low-confidence interpretations"""
        intent_result = self.intent_recognizer.extract_intent(user_command)

        if intent_result['confidence'] < self.confidence_threshold:
            # Ask for confirmation
            confirmation = self.ask_for_confirmation(user_command, intent_result)
            if not confirmation:
                return self.request_repetition()

        # Execute the action
        return self.execute_safe_action(intent_result)

    def ask_for_confirmation(self, original_command: str, intent_result: dict) -> bool:
        """Ask user to confirm interpretation"""
        confirmation_text = f"I understood you want me to {intent_result['intent']}. Is that correct?"
        self.speak(confirmation_text)

        # Listen for yes/no response
        response = self.speech_recognizer.listen_and_transcribe()
        return 'yes' in response.lower() or 'correct' in response.lower()

    def execute_safe_action(self, intent_result: dict):
        """Execute action with safety checks"""
        try:
            # Add safety validations here
            actions = self.action_mapper.map_intent_to_action(intent_result)

            # Validate actions before execution
            if self.validate_actions(actions):
                return self.execute_actions(actions)
            else:
                return self.handle_invalid_action()
        except Exception as e:
            return self.handle_execution_error(e)

    def validate_actions(self, actions: List[RobotAction]) -> bool:
        """Validate that actions are safe to execute"""
        for action in actions:
            if action.action_type == RobotActionType.MOVE_BASE:
                # Check if movement is safe
                if not self.is_path_safe(action.parameters):
                    return False
        return True

    def is_path_safe(self, params: dict) -> bool:
        """Check if planned path is safe"""
        # Implement path safety checking
        return True  # Simplified
```

## Best Practices

1. **Privacy**: Handle voice data securely, especially in personal environments
2. **Latency**: Optimize for low-latency processing for natural interaction
3. **Robustness**: Handle various accents, background noise, and speech variations
4. **Confirmation**: Ask for confirmation on critical or ambiguous commands
5. **Feedback**: Provide clear feedback about command recognition and execution
6. **Fallback**: Have fallback mechanisms when speech recognition fails

## Practical Exercise

Create a voice-to-action system:
1. Set up speech recognition using Whisper or local model
2. Implement intent recognition for basic commands
3. Map intents to robot actions
4. Integrate with ROS 2 for execution
5. Add error handling and confirmation mechanisms

## Summary

- Voice-to-action enables natural human-robot interaction
- Multiple components work together: speech recognition, NLU, action mapping
- Integration with ROS 2 enables real robot execution
- Privacy and safety are important considerations
- Error handling and confirmation improve reliability
- Advanced LLMs can provide sophisticated language understanding