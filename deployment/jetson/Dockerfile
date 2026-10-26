# Container application image. Flash JetPack 6.2.1 separately on the target.
FROM ultralytics/ultralytics@sha256:ddde06a91b658955e203499a92308ccce3ba4fdf8ebbfa8ad65a2573902b1761
COPY app/ /opt/yolo-depth/
COPY container/ /opt/yolo-depth/container/
WORKDIR /opt/yolo-depth
ENV YOLO_CONFIG_DIR=/tmp/yolo-depth-settings \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/yolo-depth-matplotlib
LABEL org.opencontainers.image.title="YOLO + Depth preview for Jetson Orin Nano" \
      org.opencontainers.image.version="0.1.0-preview-jp6.2.1-arm64" \
      ai.yolo-depth.host-jetpack="6.2.1" \
      ai.yolo-depth.board-validation="pending"
ENTRYPOINT ["python3", "/opt/yolo-depth/container/entrypoint.py"]
CMD ["--help"]
