#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import os
import tempfile
import pytest
from s3torchconnectorclient import S3Exception

from s3torchconnector._s3client import S3Client, S3ClientConfig

HELLO_WORLD_DATA = b"Hello, World!\n"
TEST_PROFILE_NAME = "test-profile"


def test_no_access_objects_without_profile(empty_directory):
    if empty_directory.profile_bucket is None:
        pytest.skip("No profile bucket configured")

    client = S3Client(
        empty_directory.region,
    )
    filename = f"{empty_directory.prefix}hello_world.txt"

    with pytest.raises(S3Exception):
        put_stream = client.put_object(
            empty_directory.profile_bucket,
            filename,
        )
        put_stream.write(HELLO_WORLD_DATA)


def test_requester_pays_access(empty_directory):
    # Note: This test requires a bucket in a different AWS account
    rp_bucket = os.getenv("CI_REQUESTER_PAYS_BUCKET")
    if not rp_bucket:
        pytest.skip("No requester-pays bucket configured")

    # Without requester_pays — cross-account access should fail with 403
    client_no_rp = S3Client(empty_directory.region)
    with pytest.raises(S3Exception):
        client_no_rp.head_object(rp_bucket, "test-object.txt")

    # With requester_pays — should succeed
    client_rp = S3Client(
        empty_directory.region,
        s3client_config=S3ClientConfig(requester_pays=True),
    )
    result = client_rp.head_object(rp_bucket, "test-object.txt")
    assert result is not None


def test_access_objects_with_profile(empty_directory):
    if empty_directory.profile_bucket is None:
        pytest.skip("No profile bucket configured")

    try:
        tmp_file = tempfile.NamedTemporaryFile()
        tmp_file.write(f"""[profile default]
aws_access_key_id = {os.getenv("AWS_ACCESS_KEY_ID")}
aws_secret_access_key = {os.getenv("AWS_SECRET_ACCESS_KEY")}
aws_session_token = {os.getenv("AWS_SESSION_TOKEN")}
 
[profile {TEST_PROFILE_NAME}]
role_arn = {empty_directory.profile_arn}
region = {empty_directory.region}
source_profile = default""".encode())
        tmp_file.flush()
        os.environ["AWS_CONFIG_FILE"] = tmp_file.name

        client = S3Client(
            empty_directory.region,
            s3client_config=S3ClientConfig(profile=TEST_PROFILE_NAME),
        )
        filename = f"{empty_directory.prefix}hello_world.txt"

        put_stream = client.put_object(
            empty_directory.profile_bucket,
            filename,
        )

        put_stream.write(HELLO_WORLD_DATA)
        put_stream.close()

        get_stream = client.get_object(
            empty_directory.profile_bucket,
            filename,
        )
        assert b"".join(get_stream) == HELLO_WORLD_DATA
    finally:
        os.environ["AWS_CONFIG_FILE"] = ""
