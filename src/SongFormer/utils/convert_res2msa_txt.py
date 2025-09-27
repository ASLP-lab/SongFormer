import json
import os
from pathlib import Path
import fire


def convert_json_to_format(json_data):
    """Convert JSON data to the specified format"""
    result = []

    # Process the start time and label for each segment
    for segment in json_data:
        start_time = segment["start"]
        label = segment["label"]
        result.append(f"{start_time:.6f} {label}")

    # Add the last end time
    if json_data:
        last_end_time = json_data[-1]["end"]
        result.append(f"{last_end_time:.6f} end")

    return "\n".join(result)


def process_json_files(input_folder, output_folder):
    """Process all JSON files in the input folder"""

    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for json_file in json_files:
        input_path = os.path.join(input_folder, json_file)

        try:
            # Read the JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert the format
            converted_data = convert_json_to_format(data)

            # Generate the output filename (replace .json with .txt)
            output_filename = json_file.replace(".json", ".txt")
            output_path = os.path.join(output_folder, output_filename)

            # Write to the output file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(converted_data)

            print(f"✓ Processed: {json_file} -> {output_filename}")

        except Exception as e:
            print(f"✗ Error processing {json_file}: {str(e)}")


def main(input_folder: str, output_folder: str):
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)

    # Process the files
    process_json_files(input_folder, output_folder)

    print("-" * 50)
    print("Processing complete!")


if __name__ == "__main__":
    fire.Fire(main)
