# Sequential Latent Variable Agent

A clean implementation for learning a world model through variational inference. The agent uses a simple MPC planner (RHE) to harness the approximate world model.

Support for:

- Gaussian RSSM
- Categorical RSSM

As the code is mostly intended to learn the world model, the purpose is to use this easily as a foundation for agents that require it as one of its components.

Although the architecture and the logic is heavily based on the RSSM proposed by Hafner et al, there are some technical differences that seem to improve the quality of the forward model for decision making:

- Action is passed on explicitly for the reward and pcont decoder
- Learn final observation


agent.py: vanilla, closely resembling the RSSMs from Hafner

agent_v2: unlike agent.py, this agent is able to learn even the terminal observation. This is a shortcoming of the agents like dreamer and planet as the instances that are appended to the buffer follow the tuple (s_t, a_t, r_t+1, d_t+1), which implies that the last observation will never be added to the buffer. To avoid this the tuple is changed to (s_t, a_t-1, r_t, d_t, i_t), where i_t corresponds to a new boolean flag which indicates whether s_t is an initial state or not. Introducing this flag then allows it to keep rssm_danijar.py intact as "noninitials" substitutes "nonterminals" as the new cleaning flag. Also this allows the agent to learn p(r_t|s_t, h_t) and not p(r_t+1|s_t, h_t) (but can also learn it too) like in the original agent.

agent_v3: considers actions explicitly, i.e. p(r_t+1|s_t,h_t,a_t)

## Buffers

full_trajectory_buffer: this buffer considers the additional flag i, to facilitate storing terminal observations.

# History

22/12/2021 - Agent v1, v2, v3 working well