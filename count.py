import boto3
import time
import json
import cv2

# https://carchi8py.com/2021/04/20/drawing-rekognition-bounding-boxes-on-images/
# https://aws.amazon.com/blogs/machine-learning/optimize-workforce-in-your-store-using-amazon-rekognition/

def create_rekognition_client():
    # Create and return a Rekognition client
    return boto3.client('rekognition', region_name='us-east-1')

def start_video_analysis(rekognition_client, bucket, video_file):
    # Start video analysis and return the JobId
    response = rekognition_client.start_label_detection(
        Video={'S3Object': {'Bucket': bucket, 'Name': video_file}},
    )
    job_id = response['JobId']
    print(f"Started analysis for video {video_file} with job id {job_id}")
    return job_id

def wait_for_job_completion(rekognition_client, job_id):
    # Wait for the Rekognition job to complete
    print("Waiting for job to complete...")
    while True:
        response = rekognition_client.get_label_detection(JobId=job_id)
        status = response['JobStatus']

        if status == 'SUCCEEDED':
            break
        elif status == 'FAILED' or status == 'PARTIAL_SUCCESS':
            print(f"Job failed or partially succeeded. Status: {status}")
            return

        time.sleep(5)  # Wait for 5 seconds before checking again

    print("Job complete")

def retrieve_video_analysis_results(rekognition_client, job_id):
    # Retrieve and return the results of the video analysis
    return rekognition_client.get_label_detection(JobId=job_id)

def write_results_to_json(results):
    # Write the video analysis results to a JSON file
    with open('data.json', 'w') as outfile:
        json.dump(results, outfile)

def read_results_from_json():
    # Read the video analysis results from a JSON file
    with open('data.json', 'r') as infile:
        return json.load(infile)

def process_results(results):
    # Process the results to count people over time
    interval_count = []

    labels = results.get('Labels', [])
    for label_detection in labels:
        label = label_detection.get('Label', {})
        if label.get('Name') == 'Person':
            instances = label.get('Instances', [])

            # Convert the timestamp from milliseconds to seconds (round to a decimal place)
            timestamp = round(label_detection.get('Timestamp') / 1000, 1)

            # Count the instances
            instance_count = len(instances)

            # Create a dictionary with timestamp and count
            data_point = {'timestamp': timestamp, 'count': instance_count}

            # Append the dictionary to the interval_count array
            interval_count.append(data_point)
    return interval_count

def print_interval_counts(interval_count):
    # Print the interval counts
    for interval in interval_count:
        print(interval)

def count_people_over_time(bucket, video_file):
    rekognition_client = create_rekognition_client()

    job_id = start_video_analysis(rekognition_client, bucket, video_file)

    wait_for_job_completion(rekognition_client, job_id)

    results = retrieve_video_analysis_results(rekognition_client, job_id)

    write_results_to_json(results)

    interval_count = process_results(results)

    print_interval_counts(interval_count)

import cv2

def add_bounding_boxes(video_path, rekognition_data, output_path):
    # Load the video file
    video_capture = cv2.VideoCapture(video_path)

    # Create a VideoWriter object to save the modified video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0

    # Variables to store the latest bounding box data
    current_bounding_boxes = []

    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Check if there are any bounding boxes for the current frame
        for label in rekognition_data['Labels']:
            if label['Label']['Name'] == 'Person':
                timestamp = label['Timestamp']
                for instance in label['Label']['Instances']:
                    instance_frame_position = int((timestamp / 1000) * fps)
                    
                    # Draw bounding box if the instance corresponds to the current frame
                    if instance_frame_position == frame_count:
                        box = instance['BoundingBox']
                        draw_bounding_box(frame, box, frame_width, frame_height)
                        current_bounding_boxes.append(box)

        # Write the modified frame to the output video
        output_video.write(frame)
        frame_count += 1

    # Release the video capture and writer
    video_capture.release()
    output_video.release()


def draw_bounding_box(frame, box, frame_width, frame_height):
    print('Drawing bounding box')
    left = int(box['Left'] * frame_width)
    top = int(box['Top'] * frame_height)
    width = int(box['Width'] * frame_width)
    height = int(box['Height'] * frame_height)
    right = left + width
    bottom = top + height
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

def add_bounding_boxes_to_video(video_path, rekognition_data, output_path):
    # Load the video file
    video_capture = cv2.VideoCapture(video_path)

    # Create a VideoWriter object to save the modified video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    timestamp_threshold = 1 / fps  # Set a threshold for timestamp comparison

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Check if there is bounding box data for the current frame
        timestamp = frame_count * 1000 / fps  # Convert frame count to milliseconds

        # Get the bounding box data for the current frame
        labels = rekognition_data.get('Labels', [])
        for label_detection in labels:
            bounding_box_timestamp = label_detection.get('Timestamp')
            # diagnost bounding_box_timestamp
            print(f"bounding_box_timestamp: {bounding_box_timestamp} - timestamp: {timestamp}")
            if abs(bounding_box_timestamp - timestamp) <= timestamp_threshold:
                # Draw bounding boxes on the frame
                instances = label_detection.get('Label', {}).get('Instances', [])
                for instance in instances:
                    box = instance.get('BoundingBox', {})
                    left = int(box.get('Left') * frame_width)
                    top = int(box.get('Top') * frame_height)
                    width = int(box.get('Width') * frame_width)
                    height = int(box.get('Height') * frame_height)
                    cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Write the modified frame to the output video
        output_video.write(frame)
        frame_count += 1

    # Release the video capture and writer
    video_capture.release()
    output_video.release()

def generate_bounding_box_video(video_file):
    results = read_results_from_json()
    interval_count = process_results(results)

    # Read the input video using OpenCV
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Iterate through the interval counts and draw bounding boxes on each frame
    for interval in interval_count:
        timestamp = float(interval.split(' ')[1])
        frame_number = int(timestamp * fps)

        # Set the frame position to the desired timestamp
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame and draw bounding boxes
        ret, frame = cap.read()
        labels = results.get('Labels', [])
        for label_detection in labels:
            label = label_detection.get('Label', {})
            if label.get('Name') == 'Person':
                instances = label.get('Instances', [])
                for instance in instances:
                    bounding_box = instance.get('BoundingBox', {})
                    left = int(bounding_box.get('Left') * width)
                    top = int(bounding_box.get('Top') * height)
                    width_bb = int(bounding_box.get('Width') * width)
                    height_bb = int(bounding_box.get('Height') * height)
                    cv2.rectangle(frame, (left, top), (left + width_bb, top + height_bb), (0, 255, 0), 2)

        # Write the frame to the output video
        output_video.write(frame)

    # Release the resources
    cap.release()
    output_video.release()

    print("Output video generated: output.mp4")

def test_process_results():
    results = read_results_from_json()
    interval_count = process_results(results)
    print_interval_counts(interval_count)

# Specify the video file name and interval in seconds
bucket = 'rekognition-video-console-demo-iad-274478531841-1687175796'
video_file = 'store.mp4'

# Call the function to count people over time
#count_people_over_time(bucket, video_file, interval_seconds)

# Test process_results function independently
test_process_results()

# Generate a video with bounding boxes
results = read_results_from_json()
add_bounding_boxes(video_file, results, 'output.mp4')