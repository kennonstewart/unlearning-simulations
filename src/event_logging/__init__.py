import logging.config, os, uuid, datetime, yaml, structlog

def init_logging():
    # --- Pick a per-run folder ---
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    # Correctly locate the base directory of the project, not just the current file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # --- Load standard logging config from YAML ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Replace placeholder path with the real run folder
    for handler in cfg["handlers"].values():
        if "filename" in handler:
            # Use the corrected log_dir variable
            handler["filename"] = handler["filename"].replace(
                "logs/current_run", log_dir
            )

    logging.config.dictConfig(cfg)

    # --- Configure structlog to process logs and pass them to standard logging ---
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,  # First, filter levels
            structlog.stdlib.add_logger_name,  # Add logger name to event dict
            structlog.stdlib.add_log_level,  # Add log level to event dict
            structlog.processors.TimeStamper(fmt="iso"),  # Add a timestamp
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # This is the critical step that renders the event dict into a JSON string
            # and passes it to the standard logging configured by your YAML file.
            structlog.stdlib.render_to_log_stream,
        ],
        # Use the standard library's logger factory
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Use the standard wrapper class for compatibility
        wrapper_class=structlog.stdlib.BoundLogger,
        # Cache logger instances for performance
        cache_logger_on_first_use=True,
    )

    return log_dir  # In case the caller wants the path