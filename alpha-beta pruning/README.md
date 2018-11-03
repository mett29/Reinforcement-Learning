# Alpha-beta pruning

"Alpha–beta pruning is a search algorithm that seeks to decrease the number of nodes that are evaluated by the minimax algorithm in its search tree. It is an adversarial search algorithm used commonly for machine playing of two-player games (Tic-tac-toe, Chess, Go, etc.). It stops completely evaluating a move when at least one possibility has been found that proves the move to be worse than a previously examined move. Such moves need not be evaluated further. When applied to a standard minimax tree, it returns the same move as minimax would, but prunes away branches that cannot possibly influence the final decision." [[Wikipedia]](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)

### Why is it better than minimax?
Minimax algorithm has a problem: the number of the game states it has to examine is exponential in the depth of the tree. 
Alpha-beta pruning doesn't eliminate the exponent, but it allows to eliminate a large part of the tree, pruning the branches which cannot possibly influence the final decision.

### Core idea
The algorithm makes use of two values, **alpha** and **beta**:

```
alpha --> value of the best (i.e. highest value) choice we have found so far at any choice point along the path for MAX;

beta --> value of the best (i.e. lowest value) choice we have found so far at any choice point along the path for MIN.
```
**alpha is initialized to -∞**
**beta is initialized to +∞**
(i.e. both players start with their worst possible score).

Whenever the maximum score that the minimizing player (i.e. the "beta" player) is assured of becomes less than the minimum score that the maximizing player (i.e., the "alpha" player) is assured of (i.e. beta ≤ alpha), the maximizing player need not consider further descendants of this node, as they will never be reached in the actual play.

#### Visual example
![example](https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/AB_pruning.svg/600px-AB_pruning.svg.png)

### References and resources
- [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)
- [Wikipedia page](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Code from which I took a cue](http://www.letscodepro.com/tic-tac-toe-minmax-alpha-beta-pruning-python/)
- [Nice visualization of the algorithm](http://will.thimbleby.net/algorithms/doku.php?id=minimax_search_with_alpha-beta_pruning)