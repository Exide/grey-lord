from environment import BBSEnvironment
import logging
from majormud import ACTIONS_BY_ID, ACTIONS_BY_COMMAND
import os
from tokenizer import GreyLordTokenizer


log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HOST = '192.168.100.2'
PORT = 23
USERNAME = 'greylord'
PASSWORD = 'password'


def main():
    pretty_commands = ', '.join(list(ACTIONS_BY_ID.values()))
    logging.info(f'Available commands: {pretty_commands}')

    tokenizer = GreyLordTokenizer()
    env = BBSEnvironment(HOST, PORT, USERNAME, PASSWORD, action_map=ACTIONS_BY_ID, tokenizer=tokenizer)
    env.reset()

    terminated = False
    try:
        while not terminated:
            try:
                command = input("").strip()
                    
                if command not in ACTIONS_BY_COMMAND:
                    logging.error(f"Unknown command: {command}")
                    continue
                    
                action_id = ACTIONS_BY_COMMAND[command]
                logging.info(f'Action ID: {action_id}')

                state, reward, terminated, truncated, info = env.step(action_id)

                logging.info(f'Reward: {reward}')
                logging.debug(f'Terminated: {terminated}, Truncated: {truncated}')
                logging.debug(f'State shape: {state.shape}')
                logging.debug(f'Info: {info}')

            except KeyboardInterrupt or EOFError:
                raise KeyboardInterrupt

            except Exception as e:
                logging.exception(f'Error encountered during this step.', exc_info=e)
                break

    finally:
        logging.info('Shutting down the environment.')
        env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('Program interrupted.')
    
    logging.info('Program ended.')

