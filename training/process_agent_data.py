#!/usr/bin/env python3
"""
Process Agent Session Data for Training

Converts agent session recordings into training data formats:
1. Continued training data (improve existing model)
2. Behavioral cloning data (train from good examples)
3. Reinforcement learning experience replay
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re


class SessionDataProcessor:
    """Processes agent session data for various training approaches."""
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.sessions = []
        self.experiences = []
        
    def load_sessions(self) -> int:
        """Load all session files from directory."""
        session_files = list(self.session_dir.glob("session_*.jsonl"))
        experience_files = list(self.session_dir.glob("experience_*.jsonl"))
        
        print(f"Found {len(session_files)} session files, {len(experience_files)} experience files")
        
        # Load session data
        for session_file in session_files:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = []
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        session_data.append(record)
                    except json.JSONDecodeError:
                        continue
                self.sessions.append({
                    'file': session_file,
                    'data': session_data
                })
        
        # Load experience data
        for exp_file in experience_files:
            with open(exp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        experience = json.loads(line.strip())
                        self.experiences.append(experience)
                    except json.JSONDecodeError:
                        continue
        
        return len(self.sessions)
    
    def create_continued_training_data(self, output_dir: Path) -> str:
        """Create training data for continued model training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sequences = []
        
        for session in self.sessions:
            sequence_data = []
            
            # Process each session chronologically
            session_data = sorted(session['data'], key=lambda x: x['timestamp'])
            
            for record in session_data:
                if record['type'] == 'server_data':
                    # Server response
                    text_data = record['data']['text_data']
                    if text_data.strip():
                        sequence_data.append(f"<|server|>{text_data}")
                
                elif record['type'] == 'ai_command':
                    # AI command
                    command = record['data']['command']
                    sequence_data.append(f"<|client|>{command}")
            
            # Join sequence and add to training data
            if sequence_data:
                full_sequence = "".join(sequence_data)
                sequences.append(full_sequence)
        
        # Save training sequences
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"agent_training_data_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for seq in sequences:
                f.write(seq + "\n\n")
        
        print(f"Created continued training data: {output_file}")
        print(f"Total sequences: {len(sequences)}")
        
        return str(output_file)
    
    def create_behavioral_cloning_data(self, output_dir: Path, success_indicators: List[str] = None) -> str:
        """Create behavioral cloning dataset from successful AI decisions."""
        if success_indicators is None:
            success_indicators = [
                "you hit", "you kill", "you gain", "you find",
                "obvious exits", "you pick up", "you get"
            ]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful_examples = []
        
        # Analyze experiences for successful outcomes
        for experience in self.experiences:
            context = experience.get('context', '')
            action = experience.get('action', '')
            
            if not context or not action:
                continue
            
            # Look for success indicators in the context following the action
            # (This is simplified - in practice you'd want more sophisticated outcome detection)
            is_successful = any(indicator in context.lower() for indicator in success_indicators)
            
            if is_successful:
                successful_examples.append({
                    'context': context,
                    'action': action,
                    'timestamp': experience.get('timestamp', 0)
                })
        
        # Save behavioral cloning data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"behavioral_cloning_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in successful_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Created behavioral cloning data: {output_file}")
        print(f"Successful examples: {len(successful_examples)}")
        
        return str(output_file)
    
    def create_rl_experience_replay(self, output_dir: Path) -> str:
        """Create experience replay buffer for reinforcement learning."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process experiences and add reward signals
        rl_experiences = []
        
        for i, experience in enumerate(self.experiences):
            # Simple reward heuristic (you'd want more sophisticated reward modeling)
            context = experience.get('context', '').lower()
            action = experience.get('action', '')
            
            # Assign rewards based on context
            reward = 0.0
            
            # Positive rewards
            if any(pos in context for pos in ['you hit', 'you kill', 'you gain', 'you find']):
                reward += 1.0
            if any(pos in context for pos in ['level up', 'you learn', 'skill increases']):
                reward += 2.0
            
            # Negative rewards
            if any(neg in context for neg in ['you die', 'you are dead', 'ouch!']):
                reward -= 2.0
            if any(neg in context for neg in ['you miss', 'blocks your attack']):
                reward -= 0.1
            
            # Neutral exploration reward
            if reward == 0.0:
                reward = 0.01  # Small exploration bonus
            
            rl_experience = {
                'state': context,
                'action': action,
                'reward': reward,
                'timestamp': experience.get('timestamp', 0),
                'session_id': experience.get('session_id', 'unknown')
            }
            
            rl_experiences.append(rl_experience)
        
        # Save RL experience replay data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rl_experience_replay_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for exp in rl_experiences:
                f.write(json.dumps(exp) + '\n')
        
        print(f"Created RL experience replay: {output_file}")
        print(f"Total experiences: {len(rl_experiences)}")
        print(f"Average reward: {sum(exp['reward'] for exp in rl_experiences) / len(rl_experiences):.3f}")
        
        return str(output_file)
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate a report on the collected data."""
        total_interactions = sum(len(session['data']) for session in self.sessions)
        ai_commands = sum(1 for session in self.sessions for record in session['data'] 
                         if record['type'] == 'ai_command')
        server_responses = sum(1 for session in self.sessions for record in session['data'] 
                              if record['type'] == 'server_data')
        
        report = {
            'sessions': len(self.sessions),
            'total_interactions': total_interactions,
            'ai_commands': ai_commands,
            'server_responses': server_responses,
            'experiences': len(self.experiences),
            'data_quality': {
                'avg_interactions_per_session': total_interactions / max(len(self.sessions), 1),
                'ai_command_ratio': ai_commands / max(total_interactions, 1),
            }
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Process agent session data for training")
    parser.add_argument("--session-dir", type=str, default="data/agent_sessions",
                       help="Directory containing session data")
    parser.add_argument("--output-dir", type=str, default="data/processed_agent_data",
                       help="Output directory for processed training data")
    parser.add_argument("--mode", type=str, choices=['all', 'continued', 'behavioral', 'rl'],
                       default='all', help="Type of training data to create")
    
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    output_dir = Path(args.output_dir)
    
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        return 1
    
    print("🔄 Processing agent session data...")
    processor = SessionDataProcessor(session_dir)
    
    # Load data
    num_sessions = processor.load_sessions()
    if num_sessions == 0:
        print("❌ No session data found")
        return 1
    
    # Generate report
    report = processor.generate_training_report()
    print(f"\n📊 Data Summary:")
    print(f"   Sessions: {report['sessions']}")
    print(f"   Total interactions: {report['total_interactions']}")
    print(f"   AI commands: {report['ai_commands']}")
    print(f"   Server responses: {report['server_responses']}")
    print(f"   Experiences: {report['experiences']}")
    print(f"   Avg interactions/session: {report['data_quality']['avg_interactions_per_session']:.1f}")
    
    # Process data based on mode
    if args.mode in ['all', 'continued']:
        print(f"\n🚀 Creating continued training data...")
        processor.create_continued_training_data(output_dir / "continued_training")
    
    if args.mode in ['all', 'behavioral']:
        print(f"\n🎯 Creating behavioral cloning data...")
        processor.create_behavioral_cloning_data(output_dir / "behavioral_cloning")
    
    if args.mode in ['all', 'rl']:
        print(f"\n🎮 Creating RL experience replay data...")
        processor.create_rl_experience_replay(output_dir / "rl_experience")
    
    print(f"\n✅ Processing complete! Check {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main()) 