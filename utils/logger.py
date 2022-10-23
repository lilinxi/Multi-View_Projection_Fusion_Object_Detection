def write_log(log_path: str, line: str):
    log = open(log_path, "a")
    log.write(line)
    log.write('\n')
    log.close()
