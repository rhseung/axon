# axon

Autograd engine and neural network framework built from scratch.

- 백엔드 추상화: NumPy (CPU) / MLX (Apple Silicon Metal) / CuPy (NVIDIA, 옵션)
- `Array` (backend native ndarray) + `Node` (graph 노드) + `Parameter` (학습 leaf) 의
  3-layer 분해. 별도 `Tensor` 래퍼는 두지 않는다.
- `Op.forward / backward` 는 backend Array 만 다뤄 numpy 스타일 수식 코드 그대로
  유지. `Op.apply` 가 Node ↔ Array 변환 plumbing 을 한 곳에서 처리.
- 마일스톤: MLP/MNIST → CNN/CIFAR → RNN/LSTM → Transformer → GPT-2 124M inference.

설계 철학과 결정의 근거는 [DESIGN.md](./DESIGN.md), 작업 계획과 마일스톤은
[PLAN.md](./PLAN.md) 를 참고.
