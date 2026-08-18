import type { Edge, Node } from '@xyflow/react'
import type { GraphEdge, GraphNode, GraphSnapshot } from './types'

export type LfgNodeData = GraphNode & { [key: string]: unknown }

export async function layoutGraph(graph: GraphSnapshot): Promise<{ nodes: Node<LfgNodeData>[]; edges: Edge[] }> {
  const columns = ['prompt', 'rro', 'agent', 'task', 'provider']
  return {
    nodes: columns.flatMap((kind, column) => {
      const items = graph.nodes.filter((node) => node.kind === kind)
      const span = Math.max(items.length - 1, 0) * 148
      return items.map((node, row) => ({
        id: node.id,
        type: 'lfg',
        position: { x: column * 282, y: row * 148 - span / 2 },
        data: node as LfgNodeData
      }))
    }),
    edges: graph.edges.map(toFlowEdge)
  }
}

function toFlowEdge(edge: GraphEdge): Edge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: 'smoothstep',
    animated: edge.state === 'committed',
    label: edge.relation,
    className: `flow-edge flow-edge--${edge.state}`
  }
}
