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
    first_response = None
    labels = []
    next_token = None

    while True:
        kwargs = {'JobId': job_id}
        if next_token:
            kwargs['NextToken'] = next_token

        response = rekognition_client.get_label_detection(**kwargs)
        if not first_response:
            first_response = response

        for label in response['Labels']:
            labels.append(label)

        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break

    first_response['Labels'] = labels
    return first_response

#def retrieve_video_analysis_results(rekognition_client, job_id):
    # Retrieve and return the results of the video analysis
#    return rekognition_client.get_label_detection(JobId=job_id)

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

def dump_results_to_json():
    rekognition_client = create_rekognition_client()
    job_id = start_video_analysis(rekognition_client, bucket, video_file)
    wait_for_job_completion(rekognition_client, job_id)
    results = retrieve_video_analysis_results(rekognition_client, job_id)
    write_results_to_json(results)

def count_people_over_time(bucket, video_file):
    rekognition_client = create_rekognition_client()
    job_id = start_video_analysis(rekognition_client, bucket, video_file)
    wait_for_job_completion(rekognition_client, job_id)
    results = retrieve_video_analysis_results(rekognition_client, job_id)
    write_results_to_json(results)
    interval_count = process_results(results)
    print_interval_counts(interval_count)

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
    current_bounding_boxes = []

    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Check if there are any bounding boxes for the current frame
        for label in rekognition_data['Labels']:
            if label['Label']['Name'] == 'Person':
                # Variables to store the latest bounding box data

                timestamp = label['Timestamp']
                instance_frame_position = int((timestamp / 1000) * fps)

                if instance_frame_position == frame_count:
                    print(f"Update bounding box for time={timestamp/1000}")
                    current_bounding_boxes = []

                    for instance in label['Label']['Instances']:
                        # Draw bounding box if the instance corresponds to the current frame
                        box = instance['BoundingBox']
                        current_bounding_boxes.append(box)

                    break
            
        if current_bounding_boxes:
            for box in current_bounding_boxes:
                draw_bounding_box(frame, box, frame_width, frame_height)

        # Write the modified frame to the output video
        output_video.write(frame)
        frame_count += 1

    # Release the video capture and writer
    video_capture.release()
    output_video.release()

def draw_bounding_box(frame, box, frame_width, frame_height):
    left = int(box['Left'] * frame_width)
    top = int(box['Top'] * frame_height)
    width = int(box['Width'] * frame_width)
    height = int(box['Height'] * frame_height)
    right = left + width
    bottom = top + height
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

def test_process_results():
    results = read_results_from_json()
    interval_count = process_results(results)
    print_interval_counts(interval_count)

# Specify the video file name and interval in seconds
bucket = 'rekognition-video-console-demo-iad-274478531841-1687175796'
video_file = 'store.mp4'

#dump_results_to_json()
# Call the function to count people over time
#count_people_over_time(bucket, video_file)

# Test process_results function independently
#test_process_results()

# Generate a video with bounding boxes
results = read_results_from_json()
add_bounding_boxes(video_file, results, 'output.mp4')