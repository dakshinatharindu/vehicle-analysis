import json
import boto3
import vehicle_count


def handler(event, context):
    s3 = boto3.client('s3')
    
    bucket = event["bucket"]
    key = event["key"]

    filepath = "/tmp/" +  key
    
    with open(filepath, 'wb') as f:
        s3.download_fileobj(bucket, key, f)
        print("successfully downloaded")

    number_of_vehicles = vehicle_count.run(filepath)
    print("successfully counted vehicles")
    
    with open("/tmp/result.mp4", 'rb') as f:
        s3.put_object( Bucket=bucket, Key="result.mp4", Body=f)
        print("successfully uploaded")
    
    return {
        'Vehicle Count': number_of_vehicles,
        'body': json.dumps("File updated successfully.")
    }


print(handler({'source':'highway.mp4'}, None))