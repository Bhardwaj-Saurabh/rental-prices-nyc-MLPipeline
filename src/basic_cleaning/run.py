#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    
    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    dataset = pd.read_csv(artifact_local_path)  

    # Perform basic cleaning    
    logger.info("Removing outliers using min and max prices")
    id_ = dataset["price"].between(args.min_price, args.max_price)  
    dataset = dataset[id_].copy()

    # Restrict long and lat between -74.25, -73.50
    idx_ = dataset['longitude'].between(-74.25, -73.50) & dataset['latitude'].between(40.5, 41.2)
    dataset = dataset[idx_].copy()

    # parse last_review to datetime
    dataset['last_review'] = pd.to_datetime(dataset['last_review'])

    # Save CSV file
    logger.info("Saving the results to CSV")
    dataset.to_csv(args.output_artifact, index=False)

    # Prepare artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    logger.info("Clean sample is prepared")
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="this steps cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the preprocessing artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the cleaning artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Name of the cleaning sample",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the preprocesses data sample",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min price for the prediction",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Max price for the prediction",
        required=True
    )

    args = parser.parse_args()

    go(args)
