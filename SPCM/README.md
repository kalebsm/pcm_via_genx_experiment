
## Overview
The Production Cost Model here is built upon code-base in GenX. GenX is an [open source](https://github.com/GenXProject/GenX/blob/main/LICENSE) capacity expansion model. Unit Commitment and Economic Dispatch constraints are updated to operate on a rolling horizon preserving state information across horizons. 

The code provided in this update on the GenX package is meant as an experiment in comparing the GenX capacity expansion model with a production cost model that operates on a rolling horizon with different lookahead foresight policies.

The GenX model was [originally developed](https://energy.mit.edu/publication/enhanced-decision-support-changing-electricity-landscape/) by 
[Jesse D. Jenkins](https://mae.princeton.edu/people/faculty/jenkins) and 
[Nestor A. Sepulveda](https://energy.mit.edu/profile/nestor-sepulveda/) at the Massachusetts Institute of Technology and is now jointly maintained by 
[a team of contributors](https://github.com/GenXProject/GenX#genx-team) at the Princeton University ZERO Lab (led by Jenkins), MIT (led by [Ruaridh MacDonald](https://energy.mit.edu/profile/ruaridh-macdonald/)), NYU (led by [Dharik Mallapragada](https://engineering.nyu.edu/faculty/dharik-mallapragada)), and Binghamton University (led by [Neha Patankar](https://www.binghamton.edu/ssie/people/profile.html?id=npatankar)).

## Acknowledgement
This work utilized the GenX open-source code base as the foundation for experimentation. We gratefully acknowledge the developers and contributors of GenX for their efforts in making the code publicly available, which enabled us to implement and evaluate our proposed modifications in a comparative study against the original implementation.

