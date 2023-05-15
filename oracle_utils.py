import pandas as pd


def evaluate_oracle_cp(model, env, num_episodes=100):
    positions = []
    velocities = []
    pole_angles = []
    pole_angle_velocities = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            positions.append(state[0])
            velocities.append(state[1])
            pole_angles.append(state[2])
            pole_angle_velocities.append(state[3])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "angle": pole_angles,
                "angle_velocity": pole_angle_velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_cp_steps(model, env, num_samples=int(3e4)):
    positions = []
    velocities = []
    pole_angles = []
    pole_angle_velocities = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            positions.append(state[0])
            velocities.append(state[1])
            pole_angles.append(state[2])
            pole_angle_velocities.append(state[3])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "angle": pole_angles,
                "angle_velocity": pole_angle_velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_mc(model, env, num_episodes=100):
    positions = []
    velocities = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            positions.append(state[0])
            velocities.append(state[1])
            actions.append(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_mcc(model, mcc_env, num_episodes=100):
    positions = []
    velocities = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = mcc_env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = mcc_env.step(action)
            positions.append(state[0])
            velocities.append(state[1])
            actions.append(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_mcc_steps(model, env, num_samples=int(3e4)):
    positions = []
    velocities = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            positions.append(state[0])
            velocities.append(state[1])
            actions.append(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_cpsu(model, env, num_episodes=100):
    positions = []
    velocities = []
    pole_cos_angles = []
    pole_sin_angles = []
    pole_angle_velocities = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            positions.append(state[0])
            velocities.append(state[1])
            pole_cos_angles.append(state[2])
            pole_sin_angles.append(state[3])
            pole_angle_velocities.append(state[4])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "angle_cos": pole_cos_angles,
                "angle_sin": pole_sin_angles,
                "angle_velocity": pole_angle_velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_ab_steps(model, env, num_samples=int(3e4)):
    theta1_cos = []
    theta1_sin = []
    theta2_cos = []
    theta2_sin = []
    theta1_dot = []
    theta2_dot = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            if not isinstance(action, int):
                action = int(action)
            theta1_cos.append(state[0])
            theta1_sin.append(state[1])
            theta2_cos.append(state[2])
            theta2_sin.append(state[3])
            theta1_dot.append(state[4])
            theta2_dot.append(state[5])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "theta1_cos": theta1_cos,
                "theta1_sin": theta1_sin,
                "theta2_cos": theta2_cos,
                "theta2_sin": theta2_sin,
                "theta1_dot": theta1_dot,
                "theta2_dot": theta2_dot,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_cpsu_steps(model, env, num_samples=int(3e4)):
    positions = []
    velocities = []
    pole_cos_angles = []
    pole_sin_angles = []
    pole_angle_velocities = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            if not isinstance(action, int):
                action = int(action)
            positions.append(state[0])
            velocities.append(state[1])
            pole_cos_angles.append(state[2])
            pole_sin_angles.append(state[3])
            pole_angle_velocities.append(state[4])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "angle_cos": pole_cos_angles,
                "angle_sin": pole_sin_angles,
                "angle_velocity": pole_angle_velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_ll_steps(model, env, num_samples=int(3e4)):
    xs = []
    ys = []
    vxs = []
    vys = []
    thetas = []
    theta_dots = []
    first_legs = []
    second_legs = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            xs.append(state[0])
            ys.append(state[1])
            vxs.append(state[2])
            vys.append(state[3])
            thetas.append(state[4])
            theta_dots.append(state[5])
            first_legs.append(state[6])
            second_legs.append(state[7])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "x": xs,
                "y": ys,
                "vx": vxs,
                "vy": vys,
                "theta": thetas,
                "theta_dot": theta_dots,
                "first_leg": first_legs,
                "second_leg": second_legs,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_mc_steps(model, env, num_samples=int(3e4)):
    positions = []
    velocities = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            positions.append(state[0])
            velocities.append(state[1])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "position": positions,
                "velocity": velocities,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_ab(model, env, num_episodes=100):
    theta1_cos = []
    theta1_sin = []
    theta2_cos = []
    theta2_sin = []
    theta1_dot = []
    theta2_dot = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            theta1_cos.append(state[0])
            theta1_sin.append(state[1])
            theta2_cos.append(state[2])
            theta2_sin.append(state[3])
            theta1_dot.append(state[4])
            theta2_dot.append(state[5])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "theta1_cos": theta1_cos,
                "theta1_sin": theta1_sin,
                "theta2_cos": theta2_cos,
                "theta2_sin": theta2_sin,
                "theta1_dot": theta1_dot,
                "theta2_dot": theta2_dot,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_pend(model, env, num_episodes=100):
    theta_cos = []
    theta_sin = []
    omegas = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            theta_cos.append(state[0])
            theta_sin.append(state[1])
            omegas.append(state[2])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "theta_cos": theta_cos,
                "theta_sin": theta_sin,
                "omega": omegas,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_pend_steps(model, env, num_samples=int(3e4)):
    theta_cos = []
    theta_sin = []
    omegas = []
    actions = []
    rewards = []
    while True:
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            theta_cos.append(state[0])
            theta_sin.append(state[1])
            omegas.append(state[2])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
        if len(actions) >= num_samples:
            break
    return (
        pd.DataFrame(
            {
                "theta_cos": theta_cos,
                "theta_sin": theta_sin,
                "omega": omegas,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_pend2d(model, env, num_episodes=100):
    thetas = []
    omegas = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            thetas.append(state[0])
            omegas.append(state[1])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "theta": thetas,
                "omega": omegas,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_ll(model, env, num_episodes=100):
    xs = []
    ys = []
    vxs = []
    vys = []
    thetas = []
    theta_dots = []
    first_legs = []
    second_legs = []
    actions = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state)
            xs.append(state[0])
            ys.append(state[1])
            vxs.append(state[2])
            vys.append(state[3])
            thetas.append(state[4])
            theta_dots.append(state[5])
            first_legs.append(state[6])
            second_legs.append(state[7])
            actions.append(action)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return (
        pd.DataFrame(
            {
                "x": xs,
                "y": ys,
                "vx": vxs,
                "vy": vys,
                "theta": thetas,
                "theta_dot": theta_dots,
                "first_leg": first_legs,
                "second_leg": second_legs,
                "action": actions,
            }
        ),
        rewards,
    )


def evaluate_oracle_fixState(model, env, states, num_episodes=100):
    observations = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset(start_space=states[i])
        done = False
        reward_sum = 0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            observations.append(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
        rewards.append(reward_sum)
    return pd.DataFrame(observations, columns=["position", "velocity"]), rewards
