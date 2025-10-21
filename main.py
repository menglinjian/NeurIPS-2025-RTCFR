import LiteEFG


class RTCFRPlusGraph(LiteEFG.Graph):
    def __init__(self, gamma=1e-10, mu=1e-3, shrink_iter=100): #default parameters
        super().__init__()
        self.timestep = 0
        self.shrink_iter = shrink_iter # shrink_iter is T_u

        # Initialization of RTCFR+
        with LiteEFG.backward(is_static=True):
            ev = 1.0 * LiteEFG.const(1, 0.0)
            # unperturbed_strategy is \sigma
            self.unperturbed_strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            # perturbed_strategy is \hat{\sigma}
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            # regret_buffer is \bm{\theta}
            self.regret_buffer = LiteEFG.const(self.action_set_size, 0.0)

            # ref_strategy is \bm{r}
            self.ref_strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            # the following three variables are used to compute \nabla \psi(\bm{r}), note that self.ref_reach_prob(I) = \nabla \psi(\bm{r})(I)
            self.ref_reach_prob = LiteEFG.const(self.action_set_size, 1.0)
            self.parent_reach_prob = LiteEFG.const(self.action_set_size, 1.0)
            self.parent_to_child_prob = LiteEFG.const(self.action_set_size, 1.0)

            self.iteration = LiteEFG.const(1, 0)
            self.mu = LiteEFG.const(1, mu)
            self.gamma = LiteEFG.const(1, gamma)
            self.alpha_I = self.gamma*self.action_set_size

        with LiteEFG.backward(color=0):
            self.iteration.inplace(self.iteration+1)
            # to compute the \hat{\bm{v}}_i^t(I) defined in (4)
            gradient = LiteEFG.aggregate(ev, aggregator="sum") + self.utility - self.mu*(self.reach_prob*self.strategy - self.ref_reach_prob*self.ref_strategy)
            # to compute the \langle \hat{\bm{v}}_i^t(I), \sigma^t_i(I) \rangle defined in (4)
            ev.inplace(LiteEFG.dot(gradient, self.unperturbed_strategy))
            # gradient - ev is the instantaneous counterfactual regret \hat{\bm{m}}_i^t(I ) defined in (4)
            self.regret_buffer.inplace(LiteEFG.maximum(self.regret_buffer + gradient - ev, 0.0))
            
            # to get \sigma^{t+1}_i(I)
            self.unperturbed_strategy.inplace(LiteEFG.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
            # to employ PCFR+ to solve the perturbed regularized EFGs, please use the following line
            # self.unperturbed_strategy.inplace(LiteEFG.normalize(self.regret_buffer + gradient - ev, p_norm=1.0, ignore_negative=True))
            # to get \hat{\sigma}^{t+1}_i(I)
            self.strategy.inplace(LiteEFG.normalize((1 - self.alpha_I)*self.unperturbed_strategy + self.gamma, p_norm=1.0, ignore_negative=True))

        # update gamma and the reference strategy profile
        with LiteEFG.backward(color=1):
            self.gamma.inplace(self.gamma * 0.5)
            self.ref_strategy.inplace(self.strategy * 1.0)
        
        with LiteEFG.forward(color=2):
            # to compute \nabla \psi(\bm{r}) after updating the reference strategy profile
            self.parent_reach_prob.inplace(LiteEFG.aggregate(self.ref_reach_prob, "sum", object="parent", player="self", padding=1))
            self.parent_to_child_prob.inplace(LiteEFG.aggregate(self.ref_strategy, "sum", object="parent", player="self", padding=1))
            self.ref_reach_prob.inplace(self.parent_reach_prob*self.parent_to_child_prob)

        
        print("===============Graph is ready for RTCFR+===============")

    def update_graph(self, env : LiteEFG.Environment) -> None:
        self.timestep += 1
        if self.timestep==1:
            env.update(self.strategy, upd_color=[2])
        if self.timestep % self.shrink_iter == 0:
            env.update(self.strategy, upd_color=[1])
            env.update(self.strategy, upd_color=[2])
            env.update(self.strategy, upd_color=[0], upd_player=1)
            env.update(self.strategy, upd_color=[0], upd_player=2)
        else:
            env.update(self.strategy, upd_color=[0], upd_player=1)
            env.update(self.strategy, upd_color=[0], upd_player=2)
        
    def current_strategy(self, type_name="last-iterate") -> LiteEFG.GraphNode:
        return self.strategy

from tqdm import tqdm
import pyspiel
import csv
import time
import numpy as np
import math
def train(graph, traverse_type, convergence_type, iter, print_freq, game_env="leduc_poker", output_strategy=False, csv_filename = "exploitability_log.csv", out_to_file=True):
    game = pyspiel.load_game(game_env)
    env = LiteEFG.OpenSpielEnv(game, traverse_type=traverse_type, regenerate=False)
    env.set_graph(graph)

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write headers
        csv_writer.writerow(['Iteration', 'Exploitability', 'Best Exploitability', 'Runing Time'])

        pbar = tqdm(total=iter)
        best_exp = 1e9
        time_start = time.time()
        for i in range(iter):
            graph.update_graph(env)
            env.update_strategy(graph.current_strategy(), update_best=(convergence_type == "best-iterate"))
                
            if (i+1) % print_freq == 0 or i==0:
                exploitability = sum(env.exploitability(graph.current_strategy(), convergence_type))
                best_exp = min(best_exp, exploitability)
                pbar.set_description(f'iterations:{i+1}, Exploitability: {exploitability:.12f}, Best: {best_exp:.12f}')
                pbar.update(print_freq)

                # Write current state to CSV
                if out_to_file:
                    time_end = time.time()
                    csv_writer.writerow([i+1, exploitability, best_exp, (time_end - time_start)/60.0])
                    csv_file.flush()

    if output_strategy:
        _, df_list = env.get_strategy(graph.current_strategy(), "avg-iterate")
        for i, df in enumerate(df_list):
            df['Infoset'] = df['Infoset'].apply(lambda x: x.replace('\n', '\\n'))
            df.to_csv("strategy_" + str(i) + ".csv", quoting=csv.QUOTE_MINIMAL, quotechar='"')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="kuhn_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument("--print_freq", type=int, default=100)

    args = parser.parse_args()
    train(RTCFRPlusGraph(), traverse_type=args.traverse_type, convergence_type="last-iterate", iter=args.iter, print_freq=args.print_freq, game_env=args.game, out_to_file=False)
