from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace, transform, array_join, make_date

def main():
    """
    Main function to run a Spark job that converts XML data 
    into structured and categorized JSON, 
    with error handling for missing or malformed fields.
    """
    # =========================================================================
    #  User Configuration
    # =========================================================================
    
    # 1. Root tag for each XML record
    xml_row_tag = 'article'

    # 2. XML field paths (use dot notation for nested tags)
    pmc_id_path = "front.article-meta.article-id"
    category_path = "front.article-meta.article-categories.subj-group"
    title_path = "front.article-meta.title-group.article-title"
    author_path = "front.article-meta.contrib-group.contrib"
    pub_date_path = "front.article-meta.pub-date"
    body_path = "body.p"

    # HDFS input and output paths
    input_path = "hdfs://namenode:9000/id2221/raw_xml_8M"
    output_path = "hdfs://namenode:9000/id2221/processed_json"

    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("XML to Structured JSON for RAG") \
        .getOrCreate()

    print("SparkSession created successfully.")
    print(f" Reading XML files from: {input_path}")

    try:
        # 1. Read XML data
        # (Best practice: use 'xml' as a format alias)
        df = spark.read \
            .format("com.databricks.spark.xml") \
            .option("rowTag", xml_row_tag) \
            .load(input_path)

        print("📘 XML data loaded. Initial Schema:")
        df.printSchema(level=3)

        # 2. Extract and transform fields
        processed_df = df.select(
            col(pmc_id_path).getItem(1).getField("_VALUE").alias("pmc_id"),
            col(category_path).getItem(0).getField("subject").alias("category"),
            col(title_path).alias("title"),
            
            transform(
                col(author_path),
                lambda x: concat_ws(
                    ", ",
                    x.getField("name").getField("surname"),
                    x.getField("name").getField("given-names")
                )
            ).alias("authors"),
            
            make_date(
                col(pub_date_path).getItem(0).getField("year"),
                col(pub_date_path).getItem(0).getField("month"),
                col(pub_date_path).getItem(0).getField("day")
            ).alias("publication_date"),
            
            # Process body text safely (handles nested or missing nodes)
            array_join(
                transform(
                    col(body_path),
                    lambda x: concat_ws(" ", x.getField("_VALUE"))
                ),
                "\n\n"
            ).alias("body_text")
        )

        # 3. Create partition column for categories
        processed_df = processed_df.withColumn(
            "category_partition",
            lower(
                regexp_replace(
                    regexp_replace(col("category").cast("string"), "[\\[\\]]", ""),  # Remove []
                    "[\\s/]+", "_"  # Replace spaces and slashes with underscores
                )
            )
        ).na.fill({"category_partition": "unknown"})

        print("Sample DataFrame after processing:")
        processed_df.show(5, truncate=True)
        print(" Final Schema before saving:")
        processed_df.printSchema()

        # 4. Save as partitioned JSON files by category
        print(f" Saving structured data to: {output_path}")
        processed_df.write \
            .partitionBy("category_partition") \
            .format("json") \
            .mode("overwrite") \
            .save(output_path)

        print(" Job completed successfully!")

    except Exception as e:
        print(f" An error occurred: {e}")

    finally:
        spark.stop()
        print(" SparkSession stopped.")


if __name__ == "__main__":
    main()
