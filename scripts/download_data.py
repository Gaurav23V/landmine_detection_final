import os
import subprocess

def download_and_extract_dataset():
  # Create data directory if it doesn't exist
  if not os.path.exists('data'):
    os.makedirs('data')

  # Change to data directory
  os.chdir('data')

  try:
    # Download and extract the dataset using the curl command
    command = 'curl -L "https://universe.roboflow.com/ds/wciLsaT5zI?key=hx3DVtIogI" > roboflow.zip && unzip roboflow.zip && rm roboflow.zip'
    subprocess.run(command, shell=True, check=True)
    print("Dataset downloaded and extracted successfully!")
  except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
  finally:
    # Change back to original directory
    os.chdir('..')

if __name__ == "__main__":
  download_and_extract_dataset()