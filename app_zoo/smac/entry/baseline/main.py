from nervex.entry import serial_pipeline

if __name__ == "__main__":
    config_path = '../smac_qmix_default_config.yaml'
    serial_pipeline(config_path, seed=0)
