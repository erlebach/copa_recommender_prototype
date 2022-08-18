# wanda is a framework to collaborate, perform parameters studies, do versioning

copa_config = {}
copa_config['use_wandb'] = False
copa_config['data'] = "../data"

if copa_config['use_wandb']:
    import wandb
