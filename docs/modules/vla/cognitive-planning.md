---
sidebar_position: 2
---

# Cognitive Planning

## Overview

Cognitive planning in Vision-Language-Action (VLA) systems represents the high-level reasoning layer that bridges natural language commands with executable robotic actions. It involves understanding complex tasks, decomposing them into manageable subtasks, and generating plans that account for the robot's capabilities, environmental constraints, and task requirements.

## Cognitive Architecture for Robotics

### The Planning Pipeline
```
Language Input → Task Understanding → World Modeling → Plan Generation → Plan Execution → Monitoring & Adaptation
```

### Key Components
- **Task Parser**: Interprets high-level goals from natural language
- **World Model**: Maintains current state and knowledge about the environment
- **Planner**: Generates sequences of actions to achieve goals
- **Executor**: Carries out planned actions with feedback
- **Monitor**: Tracks execution progress and handles exceptions

## Task Understanding and Decomposition

### Hierarchical Task Networks (HTN)
HTN planning decomposes complex tasks into simpler subtasks:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    PRIMITIVE = "primitive"
    COMPOUND = "compound"

@dataclass
class Task:
    name: str
    task_type: TaskType
    parameters: Dict[str, Any]
    subtasks: List['Task'] = None
    preconditions: List[str] = None
    effects: List[str] = None

class HTNPlanner:
    def __init__(self):
        self.domain_methods = {
            'make_coffee': self.make_coffee_method,
            'clean_table': self.clean_table_method,
            'assemble_object': self.assemble_object_method
        }

    def make_coffee_method(self) -> List[Task]:
        """Decompose 'make coffee' task into subtasks"""
        return [
            Task(
                name='navigate_to_kitchen',
                task_type=TaskType.PRIMITIVE,
                parameters={'location': 'kitchen'},
                preconditions=['robot_in_room'],
                effects=['robot_in_kitchen']
            ),
            Task(
                name='find_coffee_machine',
                task_type=TaskType.PRIMITIVE,
                parameters={'object_type': 'coffee_machine'},
                preconditions=['robot_in_kitchen'],
                effects=['coffee_machine_located']
            ),
            Task(
                name='operate_coffee_machine',
                task_type=TaskType.PRIMITIVE,
                parameters={'action': 'start_brewing'},
                preconditions=['coffee_machine_located', 'coffee_supplies_available'],
                effects=['coffee_brewing']
            ),
            Task(
                name='wait_for_coffee',
                task_type=TaskType.PRIMITIVE,
                parameters={'duration': 300},  # 5 minutes
                preconditions=['coffee_brewing'],
                effects=['coffee_ready']
            )
        ]

    def clean_table_method(self) -> List[Task]:
        """Decompose 'clean table' task into subtasks"""
        return [
            Task(
                name='navigate_to_table',
                task_type=TaskType.PRIMITIVE,
                parameters={'location': 'table'},
                preconditions=['robot_in_room'],
                effects=['robot_at_table']
            ),
            Task(
                name='scan_table',
                task_type=TaskType.PRIMITIVE,
                parameters={'sensor': 'camera'},
                preconditions=['robot_at_table'],
                effects=['table_objects_identified']
            ),
            Task(
                name='pick_up_items',
                task_type=TaskType.COMPOUND,
                parameters={},
                subtasks=self.pick_up_items_subtasks(),
                preconditions=['table_objects_identified'],
                effects=['table_items_removed']
            ),
            Task(
                name='wipe_surface',
                task_type=TaskType.PRIMITIVE,
                parameters={'tool': 'cloth', 'area': 'table_surface'},
                preconditions=['table_items_removed'],
                effects=['table_surface_clean']
            )
        ]

    def pick_up_items_subtasks(self) -> List[Task]:
        """Subtasks for picking up items from table"""
        return [
            Task(
                name='identify_graspable_object',
                task_type=TaskType.PRIMITIVE,
                parameters={'category': 'graspable'},
                preconditions=['table_objects_identified'],
                effects=['object_selected']
            ),
            Task(
                name='approach_object',
                task_type=TaskType.PRIMITIVE,
                parameters={'object_id': 'selected_object'},
                preconditions=['object_selected'],
                effects=['robot_near_object']
            ),
            Task(
                name='grasp_object',
                task_type=TaskType.PRIMITIVE,
                parameters={'object_id': 'selected_object'},
                preconditions=['robot_near_object'],
                effects=['object_grasped']
            ),
            Task(
                name='move_to_disposal',
                task_type=TaskType.PRIMITIVE,
                parameters={'destination': 'disposal_area'},
                preconditions=['object_grasped'],
                effects=['robot_at_disposal']
            ),
            Task(
                name='release_object',
                task_type=TaskType.PRIMITIVE,
                parameters={'object_id': 'selected_object'},
                preconditions=['robot_at_disposal'],
                effects=['object_released']
            )
        ]

    def decompose_task(self, task_name: str) -> Optional[List[Task]]:
        """Decompose high-level task into primitive actions"""
        if task_name in self.domain_methods:
            return self.domain_methods[task_name]()
        return None
```

## World Modeling and State Representation

### Knowledge Graph for Robotics
Representing the world state as a graph of interconnected entities:

```python
from typing import Set, Dict, List
import networkx as nx

class RobotWorldModel:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.update_timestamp = {}
        self.capabilities = self._initialize_capabilities()

    def _initialize_capabilities(self) -> Dict:
        """Initialize robot capabilities"""
        return {
            'navigation': {
                'max_speed': 1.0,  # m/s
                'min_turn_radius': 0.2,  # m
                'locations': ['kitchen', 'living_room', 'bedroom', 'office']
            },
            'manipulation': {
                'reach_range': 1.0,  # m
                'payload': 2.0,  # kg
                'gripper_types': ['parallel', 'suction']
            },
            'sensing': {
                'camera_range': 5.0,  # m
                'lidar_range': 10.0,  # m
                'resolution': 'high'
            }
        }

    def update_object_location(self, obj_id: str, location: str, confidence: float = 1.0):
        """Update object location in the world model"""
        self.graph.add_node(obj_id, type='object', location=location, confidence=confidence)
        self.graph.add_edge(obj_id, location, relation='at')
        self.update_timestamp[obj_id] = self.get_current_time()

    def update_robot_state(self, pose: Dict, battery_level: float, payload: float = 0.0):
        """Update robot state in the world model"""
        self.graph.add_node('robot',
                           type='robot',
                           pose=pose,
                           battery_level=battery_level,
                           payload=payload,
                           timestamp=self.get_current_time())

    def get_reachable_objects(self, max_distance: float = 1.0) -> List[str]:
        """Get objects within reach of the robot"""
        robot_pose = self.graph.nodes.get('robot', {}).get('pose', {'x': 0, 'y': 0})
        reachable = []

        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'object':
                obj_pose = self.get_object_pose(node_id)
                if obj_pose and self._distance(robot_pose, obj_pose) <= max_distance:
                    reachable.append(node_id)

        return reachable

    def get_object_pose(self, obj_id: str) -> Optional[Dict]:
        """Get the pose of an object"""
        location = self.graph.nodes[obj_id].get('location')
        if location:
            # In a real system, you'd look up the location's pose
            return {'x': 0, 'y': 0, 'z': 0}  # Simplified
        return None

    def _distance(self, pose1: Dict, pose2: Dict) -> float:
        """Calculate Euclidean distance between two poses"""
        dx = pose1.get('x', 0) - pose2.get('x', 0)
        dy = pose1.get('y', 0) - pose2.get('y', 0)
        dz = pose1.get('z', 0) - pose2.get('z', 0)
        return (dx**2 + dy**2 + dz**2)**0.5

    def get_current_time(self) -> float:
        """Get current time for timestamping"""
        import time
        return time.time()

    def validate_plan_preconditions(self, plan: List[Task]) -> bool:
        """Validate that preconditions are satisfied for plan execution"""
        current_state = self.get_current_state()

        for task in plan:
            if task.preconditions:
                for precondition in task.preconditions:
                    if precondition not in current_state:
                        return False
        return True

    def get_current_state(self) -> Set[str]:
        """Get current world state as a set of facts"""
        state = set()

        # Add robot state facts
        robot_node = self.graph.nodes.get('robot', {})
        battery = robot_node.get('battery_level', 100)
        if battery > 20:
            state.add('robot_battery_ok')
        if battery > 50:
            state.add('robot_battery_good')

        payload = robot_node.get('payload', 0)
        max_payload = self.capabilities['manipulation']['payload']
        if payload < max_payload:
            state.add('robot_payload_acceptable')

        # Add object location facts
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'object':
                location = attrs.get('location')
                if location:
                    state.add(f'{node_id}_at_{location}')

        return state
```

## Large Language Model Integration for Planning

### Using LLMs for Task Planning
```python
import openai
import json
from typing import Dict, List

class LLMPlanner:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_plan(self, goal: str, world_state: Dict, capabilities: Dict) -> Dict:
        """Generate a plan using LLM"""
        prompt = f"""
        You are an expert robotic task planner. Generate a detailed plan to achieve the following goal:

        GOAL: {goal}

        CURRENT WORLD STATE:
        {json.dumps(world_state, indent=2)}

        ROBOT CAPABILITIES:
        {json.dumps(capabilities, indent=2)}

        Provide your response as a JSON object with the following structure:
        {{
            "success": true/false,
            "reasoning": "Step-by-step reasoning about how to achieve the goal",
            "plan": [
                {{
                    "step": 1,
                    "action": "action_name",
                    "parameters": {{"param1": "value1", ...}},
                    "preconditions": ["condition1", "condition2", ...],
                    "expected_effects": ["effect1", "effect2", ...],
                    "confidence": 0.0-1.0
                }}
            ],
            "potential_issues": ["issue1", "issue2", ...],
            "fallback_options": ["option1", "option2", ...]
        }}

        Be specific about actions and parameters that a real robot could execute.
        Consider the robot's limitations and the current world state.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )

        try:
            plan_data = json.loads(response.choices[0].message.content.strip())
            return plan_data
        except json.JSONDecodeError:
            # Fallback: return basic plan structure
            return {
                "success": False,
                "reasoning": "Failed to parse LLM response",
                "plan": [],
                "potential_issues": ["LLM response parsing failed"],
                "fallback_options": ["Use rule-based planner"]
            }

    def refine_plan(self, original_plan: Dict, execution_feedback: Dict) -> Dict:
        """Refine plan based on execution feedback"""
        prompt = f"""
        Refine the following plan based on execution feedback:

        ORIGINAL PLAN:
        {json.dumps(original_plan, indent=2)}

        EXECUTION FEEDBACK:
        {json.dumps(execution_feedback, indent=2)}

        Provide an updated plan that addresses the issues mentioned in the feedback.
        Return the same JSON structure as before.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        try:
            refined_plan = json.loads(response.choices[0].message.content.strip())
            return refined_plan
        except json.JSONDecodeError:
            return original_plan  # Return original if refinement fails
```

## Plan Execution and Monitoring

### Execution Monitor
```python
import time
from enum import Enum
from typing import Optional

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class PlanExecutor:
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.current_plan = None
        self.current_status = ExecutionStatus.PENDING
        self.execution_log = []

    def execute_plan(self, plan: List[Dict]) -> ExecutionStatus:
        """Execute a plan step by step"""
        self.current_plan = plan
        self.current_status = ExecutionStatus.EXECUTING
        self.execution_log = []

        for step_idx, step in enumerate(plan):
            self.execution_log.append({
                'step': step_idx,
                'action': step['action'],
                'status': 'started',
                'timestamp': time.time()
            })

            try:
                success = self.execute_step(step)

                if success:
                    self.execution_log[-1].update({
                        'status': 'completed',
                        'timestamp': time.time()
                    })
                else:
                    self.execution_log[-1].update({
                        'status': 'failed',
                        'timestamp': time.time(),
                        'error': 'Action execution failed'
                    })
                    self.current_status = ExecutionStatus.FAILED
                    return self.current_status

            except Exception as e:
                self.execution_log[-1].update({
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e)
                })
                self.current_status = ExecutionStatus.FAILED
                return self.current_status

        self.current_status = ExecutionStatus.SUCCESS
        return self.current_status

    def execute_step(self, step: Dict) -> bool:
        """Execute a single step of the plan"""
        action = step['action']
        params = step['parameters']

        # Map action name to actual robot command
        if action == 'navigate_to':
            return self.robot_interface.navigate_to(params['location'])
        elif action == 'pick_object':
            return self.robot_interface.pick_object(params['object_id'])
        elif action == 'place_object':
            return self.robot_interface.place_object(params['object_id'], params['location'])
        elif action == 'detect_object':
            return self.robot_interface.detect_object(params['object_type'])
        elif action == 'speak':
            return self.robot_interface.speak(params['text'])
        else:
            # Unknown action
            return False

    def get_execution_feedback(self) -> Dict:
        """Get feedback about plan execution"""
        return {
            'status': self.current_status.value,
            'completed_steps': len([log for log in self.execution_log if log['status'] == 'completed']),
            'total_steps': len(self.current_plan) if self.current_plan else 0,
            'execution_log': self.execution_log,
            'success_rate': len([log for log in self.execution_log if log['status'] == 'completed']) / len(self.execution_log) if self.execution_log else 0
        }
```

## Reactive Planning and Adaptation

### Handling Plan Failures
```python
class ReactivePlanner:
    def __init__(self):
        self.recovery_strategies = {
            'navigation_failure': self.handle_navigation_failure,
            'grasping_failure': self.handle_grasping_failure,
            'object_not_found': self.handle_object_not_found,
            'collision_detected': self.handle_collision
        }

    def handle_navigation_failure(self, failed_step: Dict, world_state: Dict) -> Optional[List[Dict]]:
        """Handle navigation failure by finding alternative path"""
        # Try alternative navigation method
        alternative_paths = self.find_alternative_paths(
            failed_step['parameters']['location'],
            world_state
        )

        if alternative_paths:
            # Generate new plan with alternative path
            new_plan = self.generate_navigation_plan(alternative_paths[0])
            return new_plan

        # If no alternatives, try to understand why navigation failed
        obstacle_info = self.analyze_environment(failed_step['parameters']['location'])
        if 'temporary_obstacle' in obstacle_info:
            # Wait and retry
            return [{
                'action': 'wait',
                'parameters': {'duration': 30},  # Wait 30 seconds
                'recovery_of': failed_step
            }, failed_step]  # Then retry original step

        return None

    def handle_grasping_failure(self, failed_step: Dict, world_state: Dict) -> Optional[List[Dict]]:
        """Handle grasping failure"""
        object_id = failed_step['parameters']['object_id']

        # Check object properties
        object_props = self.get_object_properties(object_id, world_state)

        if object_props.get('size') == 'large':
            # Try two-handed grasp or alternative approach
            return [{
                'action': 'approach_object',
                'parameters': {
                    'object_id': object_id,
                    'approach_type': 'two_handed'
                }
            }, {
                'action': 'grasp_object',
                'parameters': {
                    'object_id': object_id,
                    'grasp_type': 'power_grasp'
                }
            }]
        elif object_props.get('fragility') == 'fragile':
            # Use gentle grasp
            return [{
                'action': 'grasp_object',
                'parameters': {
                    'object_id': object_id,
                    'grasp_type': 'precision_grasp',
                    'force': 'gentle'
                }
            }]
        else:
            # Object might have moved, re-locate it
            return [{
                'action': 'locate_object',
                'parameters': {'object_type': object_props.get('type')},
                'recovery_of': failed_step
            }, failed_step]  # Then retry original step

    def handle_object_not_found(self, failed_step: Dict, world_state: Dict) -> Optional[List[Dict]]:
        """Handle case where expected object is not found"""
        object_type = failed_step['parameters'].get('object_type') or failed_step['parameters'].get('object_id')

        # Expand search area
        search_expansion = self.expand_search_area(
            failed_step['parameters'].get('search_location', 'current_area'),
            world_state
        )

        # Generate plan to search in expanded area
        search_plan = []
        for location in search_expansion:
            search_plan.extend([
                {
                    'action': 'navigate_to',
                    'parameters': {'location': location}
                },
                {
                    'action': 'detect_object',
                    'parameters': {'object_type': object_type}
                }
            ])

        # If found, continue with original intent
        original_step = failed_step.copy()
        search_plan.append(original_step)

        return search_plan

    def find_alternative_paths(self, destination: str, world_state: Dict) -> List[str]:
        """Find alternative paths to destination"""
        # This would typically interface with a path planning system
        # For now, return a list of possible alternative locations
        return [destination + '_alternative_1', destination + '_alternative_2']

    def generate_navigation_plan(self, path: str) -> List[Dict]:
        """Generate navigation plan for a given path"""
        return [{
            'action': 'navigate_to',
            'parameters': {'location': path}
        }]

    def analyze_environment(self, location: str) -> Dict:
        """Analyze environment at location for obstacles"""
        # In a real system, this would use sensors to analyze the environment
        return {'temporary_obstacle': False}

    def get_object_properties(self, object_id: str, world_state: Dict) -> Dict:
        """Get properties of an object"""
        # In a real system, this would query the world model
        return {'size': 'medium', 'fragility': 'normal', 'type': 'unknown'}

    def expand_search_area(self, base_location: str, world_state: Dict) -> List[str]:
        """Expand search area from base location"""
        # Return adjacent locations or areas to search
        return [base_location + '_adjacent_1', base_location + '_adjacent_2']
```

## Integration Example: Complete Cognitive Planning System

### Putting It All Together
```python
class CognitivePlanner:
    def __init__(self, llm_api_key: str):
        self.world_model = RobotWorldModel()
        self.hierarchical_planner = HTNPlanner()
        self.llm_planner = LLMPlanner(llm_api_key)
        self.executor = PlanExecutor(robot_interface=None)  # Will be set later
        self.reactive_planner = ReactivePlanner()

    def plan_and_execute(self, goal: str) -> Dict:
        """Plan and execute a goal with full cognitive capabilities"""

        # 1. Update world model with current state
        current_state = self.world_model.get_current_state()

        # 2. Generate initial plan using LLM
        plan_response = self.llm_planner.generate_plan(
            goal=goal,
            world_state=current_state,
            capabilities=self.world_model.capabilities
        )

        if not plan_response['success']:
            # Fallback to hierarchical planner
            htn_plan = self.hierarchical_planner.decompose_task(goal)
            if htn_plan:
                plan = self.convert_to_executable_plan(htn_plan)
            else:
                return {'success': False, 'reason': 'Could not generate plan for goal'}
        else:
            plan = plan_response['plan']

        # 3. Validate plan preconditions
        if not self.world_model.validate_plan_preconditions(plan):
            return {'success': False, 'reason': 'Plan preconditions not satisfied'}

        # 4. Execute plan with monitoring
        execution_status = self.executor.execute_plan(plan)

        # 5. Handle failures reactively
        if execution_status == ExecutionStatus.FAILED:
            feedback = self.executor.get_execution_feedback()
            recovery_plan = self.handle_execution_failure(feedback, goal)

            if recovery_plan:
                recovery_status = self.executor.execute_plan(recovery_plan)
                return {
                    'success': recovery_status == ExecutionStatus.SUCCESS,
                    'original_plan': plan,
                    'recovery_plan': recovery_plan,
                    'execution_log': self.executor.get_execution_feedback()
                }

        return {
            'success': execution_status == ExecutionStatus.SUCCESS,
            'plan': plan,
            'execution_log': self.executor.get_execution_feedback()
        }

    def convert_to_executable_plan(self, htn_plan: List[Task]) -> List[Dict]:
        """Convert HTN plan to executable format"""
        executable_plan = []

        for task in htn_plan:
            executable_plan.append({
                'action': task.name,
                'parameters': task.parameters,
                'preconditions': task.preconditions or [],
                'expected_effects': task.effects or []
            })

        return executable_plan

    def handle_execution_failure(self, feedback: Dict, original_goal: str) -> Optional[List[Dict]]:
        """Handle plan execution failure"""
        failed_step = None
        for log_entry in feedback['execution_log']:
            if log_entry['status'] in ['failed', 'error']:
                # Find the original plan step that corresponds to this failure
                if hasattr(self.executor, 'current_plan') and self.executor.current_plan:
                    step_idx = log_entry['step']
                    if step_idx < len(self.executor.current_plan):
                        failed_step = self.executor.current_plan[step_idx]
                        break

        if failed_step:
            # Determine failure type and apply recovery strategy
            failure_type = self.determine_failure_type(failed_step, feedback)
            recovery_plan = self.reactive_planner.recovery_strategies.get(
                failure_type, lambda x, y: None
            )(failed_step, self.world_model.get_current_state())

            return recovery_plan

        return None

    def determine_failure_type(self, failed_step: Dict, feedback: Dict) -> str:
        """Determine the type of failure that occurred"""
        action = failed_step.get('action', '')

        if 'navigate' in action.lower():
            return 'navigation_failure'
        elif 'grasp' in action.lower() or 'pick' in action.lower():
            return 'grasping_failure'
        elif 'detect' in action.lower() or 'find' in action.lower():
            return 'object_not_found'
        elif 'collision' in str(feedback.get('error', '')):
            return 'collision_detected'
        else:
            return 'general_failure'

def main():
    """Example usage of cognitive planning system"""
    # Initialize the cognitive planner
    planner = CognitivePlanner(api_key="your-openai-api-key")

    # Example goal
    goal = "Go to the kitchen, find a red cup on the counter, pick it up, and bring it to me in the living room"

    # Plan and execute
    result = planner.plan_and_execute(goal)

    print(f"Execution result: {result['success']}")
    print(f"Execution log: {result['execution_log']}")

if __name__ == "__main__":
    main()
```

## Best Practices for Cognitive Planning

1. **Modularity**: Keep planning components modular and replaceable
2. **Robustness**: Handle failures gracefully with recovery strategies
3. **Efficiency**: Balance planning thoroughness with real-time constraints
4. **Learning**: Adapt planning strategies based on execution experience
5. **Human-in-the-loop**: Allow human intervention when needed
6. **Safety**: Ensure all plans are safe before execution

## Challenges and Future Directions

### Current Challenges
- **Real-time Performance**: Balancing sophisticated reasoning with real-time execution
- **Uncertainty Handling**: Dealing with uncertain perceptions and outcomes
- **Scalability**: Managing complex, long-horizon tasks
- **Multi-modal Integration**: Coherently combining vision, language, and action

### Emerging Approaches
- **Neuro-Symbolic Planning**: Combining neural networks with symbolic reasoning
- **Learning from Demonstration**: Acquiring planning knowledge from human demonstrations
- **Foundation Models**: Using large pre-trained models for generalizable planning
- **Collaborative Planning**: Multi-robot coordination and human-robot collaboration

## Practical Exercise

Create a cognitive planning system:
1. Implement a basic world model with state tracking
2. Create a simple HTN planner for common tasks
3. Integrate with an LLM for flexible planning
4. Add execution monitoring and recovery
5. Test with complex multi-step tasks

## Summary

- Cognitive planning bridges high-level goals with low-level actions
- Hierarchical task networks decompose complex tasks
- World models maintain state and support reasoning
- LLMs enable flexible, natural language-based planning
- Reactive planning handles execution failures
- Safety and robustness are paramount in cognitive systems