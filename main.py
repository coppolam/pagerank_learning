import consensus
import graphless_optimization

f, policy, des, H, A, E = consensus.run(1)
graphless_optimization.main(policy, des, H, A, E)


