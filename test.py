import logging
from tqdm import tqdm
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main_task():
    for i in tqdm(range(100), desc="Processing"):
        time.sleep(0.1)  # Simulate a task by sleeping for 0.1 seconds
        if i % 10 == 0:  # Log progress every 10 iterations
            tqdm.write(f'{logging.info(f"Iteration {i} completed.")}')

def main():
    logging.info('Task started.')
    main_task()
    logging.info('Task completed.')

if __name__ == "__main__":
    main()
