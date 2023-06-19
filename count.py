import boto3
import time
import json

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

            # convert the timestamp from milliseconds to seconds (round to a deciml place)
            timestamp = round(label_detection.get('Timestamp') / 1000, 1)

            # Count the instances
            instance_count = len(instances)
            interval_count.append("time: " + str(timestamp) + " seconds, count: " + str(instance_count))
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

def test_process_results():
    results = read_results_from_json()
    interval_count = process_results(results)
    print_interval_counts(interval_count)

# Specify the video file name and interval in seconds
bucket = 'rekognition-video-console-demo-iad-274478531841-1687175796'
video_file = 'store.mp4'
interval_seconds = 10

# Call the function to count people over time
#count_people_over_time(bucket, video_file, interval_seconds)

# Test process_results function independently
test_process_results()
