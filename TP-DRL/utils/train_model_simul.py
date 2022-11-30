import os, sys, time, shutil, yaml, pathlib

from ai.src.run.constants import OUTPUT_DIR, EXPERIMENT_CONFIG_FILENAME, ENVIRONMENT_CONFIG_FILENAME

from stable_baselines3.common.callbacks import CheckpointCallback


def train_model(cfg_file: str, training_dir=None, simul_port=20000, ):
    """
    Train a DRL network on a given environment as described in cfg_file,
    and save the results and config for reproductibility

    Parameters:
        param cfg_file: path of the yaml configuration file
        training_dir:   if None, a training directory will be created, else the parameter is used
                        as the training directory location.
        simul_port:     some simulators (copeliasim for exemple) require a connexion port numebr
    
    Return: None
    """

    # import parameters from config files
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f.read())

    try:
        environment  = cfg['env']['environment']
        cfg_file_env = cfg['env']['cfg_env']

        out_path     = cfg['train']['output_model_path']
        agent_type   = cfg['train']['agent_type']
        policy       = cfg['train']['policy']
        tot_steps    = cfg['train']['total_timesteps']
        save_freq    = cfg['train']['save_freq']
        nb_steps     = cfg['train']['n_steps']
        b_size       = cfg['train']['batch_size']
        nb_epochs    = cfg['train']['n_epochs']
        seed         = cfg['train']['seed']
        headless     = cfg['train']['headless']
        
    except:
        raise RuntimeError("Parameters missing in file <{cfg_file}>")
    
    with open(cfg_file_env, 'r') as f:
        cfg_env = yaml.safe_load(f.read())

    # import agent
    if  agent_type == 'PPO':
        from stable_baselines3 import PPO as agent
    else:
        raise Exception("Agent <{agent_type}> not implemented")

    if training_dir is None:
        experiment_time = time.localtime()
        # prepare directory for output
        experiment_id = "_".join([environment, agent_type,
                                time.strftime("%y-%m-%d_%H-%M-%S", experiment_time)])
        training_dir = pathlib.Path(out_path)/experiment_id
    
    training_dir = pathlib.Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    

    # Create env for training

    if environment == 'RoboticArm_2DOF_PyBullet':
    
        robot_urdf = cfg['env']['urdf']
        
        veloc     = cfg_env['velocity']
        dt        = cfg_env['dt']
        version   = cfg_env['version']
        reward    = cfg_env['reward']   
                

        env = RoboticArm_2DOF_PyBullet(robot_urdf, 
                                       target_urdf,
                                       dt=dt,
                                       reward=reward,
                                       target_pos = (0.5, 0.5),
                                       clip_action: bool = False,
                                       veloc_max: float= 1,
                                       headless: bool = False,
                                       verbose: int = 1):
        
        shutil.copyfile('balancing_robot/python/rewards.py', training_dir/'rewards.py')

    else:
        raise Exception("Not implemented environment: <{environment}>")

    
    # copy precious files in experiment_dir
    shutil.copyfile(cfg_file, training_dir/EXPERIMENT_CONFIG_FILENAME)
    shutil.copyfile(cfg_file_env, training_dir/ENVIRONMENT_CONFIG_FILENAME)
    
    # prepare agent for training
    model = agent(policy, 
                  env, 
                  n_epochs=nb_epochs,
                  n_steps=nb_steps,
                  batch_size=b_size,
                  seed=seed,
                  tensorboard_log=training_dir,
                  verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=save_freq, 
                                             save_path=training_dir/'ZIP')

    # train agent
    t0 = time.time()

    model.learn(total_timesteps=tot_steps, 
                callback=checkpoint_callback)
    
    t = int(time.time()-t0)
    h = int(t//3600)
    m = int((t - h*3600)//60)
    print(f"Training elapsed time: {h:02d}h {m:02d}m")
  
    # save trained agent
    target_zip = os.path.join(training_dir, 'ZIP', 'model.zip')
    print(f"saving trained model in <{target_zip}>")
    model.save(target_zip)
    env.close()

    os.system(f"cd {out_path} && ln -s -f {os.path.basename(training_dir)} last")

    return training_dir
    

if __name__ == "__main__":
    # bloc main
    import argparse, sys
    parser = argparse.ArgumentParser()
    
    group  = parser.add_mutually_exclusive_group()
    group.add_argument('--vehicule', action="store", dest='vehicule', 
                         help="keyword in (cartpole, balancingrobot, miniapterros)")
    
    group.add_argument('--config', action="store", dest='config', 
                         help="relative path name of the file '..._ppo_<SIMUL>.yaml'")

    parser.add_argument('--port', action="store", dest='port', default="20000", type=int, 
                         help="coppeliasim port, default is 20000")
    
    parser.add_argument('--traindir', action="store", dest='traindir', 
                         help="Optional, the relative pathname of the training directory")
    
    parser.add_argument('--simulator', action="store", dest='simulator', type=str, 
                         help="Which simulator use in (gym, copsim, pybullet)")
    
    args = parser.parse_args()
    assert args.simulator in ("gym", "copsim", "pybullet")
    
    if args.vehicule:
        config_path = f"ai/config/{args.vehicule}_ppo_{args.simulator}.yaml"        
    elif args.config:
        config_path = args.config
    else:
        parser.print_help()
        sys.exit(1)

    traindir = None
    if args.traindir: traindir = args.traindir

    print(f"running: train_model('{config_path}', simul_port={args.port}, training_dir='{traindir}')")

    train_model(config_path, simul_port=args.port, training_dir=traindir)
