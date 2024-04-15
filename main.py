import matplotlib.pyplot as plt
from gridCreation import GridWorldSimulation

def plot_rewards(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Cumulative Rewards')
    plt.xlabel('Simulation Step')
    plt.ylabel('Cumulative Rewards')
    plt.title('Agent Learning Progress Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reset_statistics(reset_counts):
    plt.figure(figsize=(10, 5))
    plt.plot(reset_counts, marker='o', linestyle='-')
    plt.title('Number of Actions Between Resets')
    plt.xlabel('Reset Index')
    plt.ylabel('Number of Actions')
    plt.grid(True)
    plt.show()

def run_and_plot_simulation(total_steps, initial_policy='PRandom', subsequent_policy='PGreedy', change_step=500, seed=None):
    # Initialize and run the simulation with dynamic policy switching
    simulation = GridWorldSimulation(total_steps, initial_policy, subsequent_policy, change_step, seed)
    total_rewards, action_count, reset_counts = simulation.run()
    print(f"Total Rewards: {total_rewards}, Total Actions Taken: {action_count}")

    # Plot the cumulative rewards history and reset statistics
    plot_rewards(simulation.reward_history)
    plot_reset_statistics(reset_counts)

# Example of running the simulation with specific policies
'''
Expirement 1 with first seed
'''
# run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PRandom', change_step=500, seed=42)
run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PGreedy', change_step=500, seed=42)
# run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PExploit', change_step=500, seed=42)


'''
Expirement 1 with second seed
'''
# run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PRandom', change_step=500, seed=43)
# run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PGreedy', change_step=500, seed=43)
# run_and_plot_simulation(9000, initial_policy='PRandom', subsequent_policy='PExploit', change_step=500, seed=43)