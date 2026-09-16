FROM tensorflow/serving:2.11.0

# Copy the saved model to the container
COPY kendrickfff-pipeline/serving_model /models/churn-model
COPY monitoring.config /models/monitoring.config

# Set environment variables for TF Serving
ENV MODEL_NAME=churn-model
ENV MODEL_BASE_PATH=/models/churn-model

# Expose ports: 8501 for REST API, 8500 for gRPC
EXPOSE 8501
EXPOSE 8500

# Enable Prometheus monitoring metrics
ENV TF_CPP_MIN_LOG_LEVEL=0

# Start TF Serving with REST API
ENTRYPOINT ["tensorflow_model_server"]
CMD ["--rest_api_port=8501", \
     "--model_name=churn-model", \
     "--model_base_path=/models/churn-model"]
