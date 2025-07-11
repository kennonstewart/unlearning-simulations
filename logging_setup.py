import logging.config, os, uuid, datetime, yaml, structlog

def init_logging():
    # --- pick a per-run folder ---
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.join(os.path.dirname(__file__), "logs", run_id)
    os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(os.path.dirname(__file__), "logging_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    # Replace placeholder path with the real run folder
    for handler in cfg["handlers"].values():
        if "filename" in handler:
            handler["filename"] = handler["filename"].replace(
                "logs/current_run/", f"{base_dir}/"
            )

    logging.config.dictConfig(cfg)

    # Configure structlog to funnel through std lib logging
    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )

    return base_dir  # in case caller wants the path