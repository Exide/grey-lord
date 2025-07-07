import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer

from agent import create_agent
from environment import BBSEnvironment
from majormud import ACTIONS_BY_ID


# Default configuration
DEFAULT_CONFIG = {
    "host": "192.168.100.2",
    "port": 23,
    "username": "greylord",
    "password": "password",
    "agent_type": "dqn",
    "episodes": 1000,
    "max_steps": 1000,
    "save_interval": 100,
    "log_interval": 10,
    "model_dir": "models",
    "tokenizer_name": "distilbert-base-uncased",
    "max_observations": 4,
    "observation_window": 1024,
    "agent_config": {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "buffer_size": 10000,
        "batch_size": 32,
        "update_frequency": 4
    }
}


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bbs_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Merge with defaults
    merged_config = DEFAULT_CONFIG.copy()
    merged_config.update(config)
    
    return merged_config


def save_config(config, config_path):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_environment(config):
    """Create and return the BBS environment."""
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    env = BBSEnvironment(
        host=config['host'],
        port=config['port'],
        username=config['username'],
        password=config['password'],
        action_map=ACTIONS_BY_ID,
        tokenizer=tokenizer,
        max_observations=config['max_observations'],
        observation_window=config['observation_window']
    )
    
    return env


def train_agent(config, resume_from=None):
    """Train an agent in the BBS environment."""
    logging.info("Starting training...")
    
    # Create environment
    env = create_environment(config)
    
    # Create agent
    agent = create_agent(
        config['agent_type'],
        action_size=len(ACTIONS_BY_ID),
        observation_shape=(config['max_observations'], config['observation_window']),
        **config['agent_config']
    )
    
    # Load existing model if resuming
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        logging.info(f"Resumed training from {resume_from}")
    
    # Ensure model directory exists
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Training loop
    total_rewards = []
    episode_steps = []
    
    for episode in range(config['episodes']):
        state = env.reset()[0]  # Get initial state
        total_reward = 0
        steps = 0
        
        for step in range(config['max_steps']):
            # Agent chooses action
            action = agent.act(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Agent learns
            agent.update(state, action, reward, next_state, done)
            
            # Update state and tracking
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Logging
        if episode % config['log_interval'] == 0:
            avg_reward = np.mean(total_rewards[-config['log_interval']:])
            avg_steps = np.mean(episode_steps[-config['log_interval']:])
            epsilon = getattr(agent, 'epsilon', 'N/A')
            
            logging.info(f"Episode {episode:4d} | "
                        f"Avg Reward: {avg_reward:8.2f} | "
                        f"Avg Steps: {avg_steps:6.1f} | "
                        f"Epsilon: {epsilon}")
        
        # Save model periodically
        if episode % config['save_interval'] == 0 and episode > 0:
            model_path = os.path.join(config['model_dir'], f"model_episode_{episode}.pt")
            agent.save(model_path)
            logging.info(f"Saved model to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(config['model_dir'], "final_model.pt")
    agent.save(final_model_path)
    logging.info(f"Training completed. Final model saved to {final_model_path}")
    
    env.close()
    return agent


def evaluate_agent(config, model_path, episodes=10):
    """Evaluate a trained agent."""
    logging.info(f"Evaluating agent from {model_path}")
    
    # Create environment
    env = create_environment(config)
    
    # Create agent
    agent = create_agent(
        config['agent_type'],
        action_size=len(ACTIONS_BY_ID),
        observation_shape=(config['max_observations'], config['observation_window']),
        **config['agent_config']
    )
    
    # Load model
    agent.load(model_path)
    
    # Disable exploration for evaluation
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0
    
    # Evaluation loop
    total_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        steps = 0
        
        for step in range(config['max_steps']):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        episode_steps.append(steps)
        
        logging.info(f"Episode {episode + 1:2d} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Steps: {steps:4d}")
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(episode_steps)
    
    logging.info(f"Evaluation completed:")
    logging.info(f"Average Reward: {avg_reward:.2f}")
    logging.info(f"Average Steps: {avg_steps:.1f}")
    
    env.close()
    return avg_reward, avg_steps


def interactive_mode(config, model_path=None):
    """Interactive mode for manual testing."""
    logging.info("Starting interactive mode...")
    
    # Create environment
    env = create_environment(config)
    
    # Create agent if model provided
    agent = None
    if model_path:
        agent = create_agent(
            config['agent_type'],
            action_size=len(ACTIONS_BY_ID),
            observation_shape=(config['max_observations'], config['observation_window']),
            **config['agent_config']
        )
        agent.load(model_path)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = 0.0  # No exploration in interactive mode
    
    # Create reverse action map for manual commands
    reverse_action_map = {v: k for k, v in ACTIONS_BY_ID.items()}
    
    state = env.reset()[0]
    
    logging.info("Interactive mode started. Available commands:")
    logging.info(f"Commands: {', '.join(ACTIONS_BY_ID.values())}")
    logging.info("Special commands: 'quit', 'agent' (if agent loaded), 'actions'")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'actions':
                print("Available actions:")
                for action_id, action_cmd in ACTIONS_BY_ID.items():
                    print(f"  {action_id}: {action_cmd}")
                continue
            elif command == 'agent' and agent:
                action = agent.act(state)
                print(f"Agent suggests action: {action} ({ACTIONS_BY_ID[action]})")
                continue
            
            # Try to execute command
            if command in reverse_action_map:
                action = reverse_action_map[command]
            else:
                try:
                    action = int(command)
                    if action not in ACTIONS_BY_ID:
                        print(f"Invalid action ID: {action}")
                        continue
                except ValueError:
                    print(f"Unknown command: {command}")
                    continue
            
            # Execute action
            state, reward, terminated, truncated, _ = env.step(action)
            print(f"Action: {ACTIONS_BY_ID[action]} | Reward: {reward}")
            
            if terminated or truncated:
                print("Episode ended. Resetting environment...")
                state = env.reset()[0]
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            logging.error(f"Error in interactive mode: {e}")
            continue
    
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BBS Agent Training and Evaluation')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'interactive'],
                       default='train', help='Mode to run')
    parser.add_argument('--model', type=str, help='Model path for evaluation or interactive mode')
    parser.add_argument('--resume', type=str, help='Resume training from model path')
    parser.add_argument('--episodes', type=int, help='Number of episodes (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        config['episodes'] = args.episodes
    
    logging.info(f"Running in {args.mode} mode")
    logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        if args.mode == 'train':
            train_agent(config, resume_from=args.resume)
        elif args.mode == 'eval':
            if not args.model:
                logging.error("Model path required for evaluation mode")
                sys.exit(1)
            evaluate_agent(config, args.model)
        elif args.mode == 'interactive':
            interactive_mode(config, args.model)
    
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
