import logging.config
import os
import uuid
import datetime
import yaml
import structlog

def init_logging():
    # --- Pick a per-run folder ---
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # --- Load standard logging config from YAML ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for handler in cfg["handlers"].values():
        if "filename" in handler:
            handler["filename"] = handler["filename"].replace(
                "logs/current_run", log_dir
            )

    # This now configures the handlers AND the structlog formatter
    logging.config.dictConfig(cfg)

    # --- Configure structlog to use the new ProcessorFormatter ---
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # This is the key change: the ProcessorFormatter defined in the YAML
            # will apply its own processors, including the JSONRenderer.
            structlog.stdlib.render_to_log_kwargs,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    return log_dir