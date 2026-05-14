import json
import re
import os

def extract_data(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # The text has formats like "t = 1 0.1701" or "t = 1 00 0.0077" due to pdf/docx extraction artifacts
    # We will find all occurrences of "t = \d+\s*\d* \d+\.\d+" and clean them up
    # A robust regex for t = <number> <value>
    pattern = r't\s*=\s*(\d+)(?:\s*(\d+))?\s+(\d+\.\d+)'
    
    matches = re.findall(pattern, content)
    
    data_points = []
    for m in matches:
        if m[1]: # if it found a space-separated number part like '1 00'
            t = int(m[0] + m[1])
        else:
            t = int(m[0])
        val = float(m[2])
        data_points.append((t, val))
        
    # Handle possible duplicates or out of order
    data_points = sorted(list(set(data_points)), key=lambda x: x[0])
    
    training_data = {str(t): val for t, val in data_points if 1 <= t <= 100}
    test_data = {str(t): val for t, val in data_points if 101 <= t <= 120}
    
    # Save to JSON files
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=4)
        
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=4)
        
    print(f"Extracted {len(training_data)} training points and {len(test_data)} test points.")

if __name__ == "__main__":
    # Ensure script is run in the directory containing PMC3.txt
    txt_path = 'PMC3.txt'
    if not os.path.exists(txt_path):
        txt_path = os.path.join(os.path.dirname(__file__), 'PMC3.txt')
    extract_data(txt_path)
