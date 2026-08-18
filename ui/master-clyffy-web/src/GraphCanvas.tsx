import { memo } from 'react'
import {
  Background, Controls, Handle, MiniMap, Position, ReactFlow, ReactFlowProvider,
  type Edge, type Node, type NodeProps
} from '@xyflow/react'
import { motion } from 'motion/react'
import '@xyflow/react/dist/style.css'

import type { LfgNodeData } from './graph'
import type { GraphNode } from './types'

type Props = { nodes: Node<LfgNodeData>[]; edges: Edge[]; onSelect: (node: GraphNode) => void }

export default function GraphCanvas({ nodes, edges, onSelect }: Props) {
  return <ReactFlowProvider><ReactFlow
    nodes={nodes} edges={edges} nodeTypes={{ lfg: LfgNode }}
    onNodeClick={(_, node) => onSelect(node.data as GraphNode)}
    fitView fitViewOptions={{ padding: 0.24 }} minZoom={0.35} maxZoom={1.8}
    proOptions={{ hideAttribution: true }}
  ><Background gap={28} size={1} /><MiniMap pannable zoomable nodeColor={miniMapColor} /><Controls showInteractive={false} /></ReactFlow></ReactFlowProvider>
}

const LfgNode = memo(function LfgNode({ data, selected }: NodeProps) {
  const node = data as LfgNodeData
  return <motion.article className={`lfg-node tone-${node.tone} kind-${node.kind} ${selected ? 'is-selected' : ''}`} initial={{ opacity: 0, scale: 0.88 }} animate={{ opacity: 1, scale: 1 }}>
    <Handle type="target" position={Position.Left} className="node-handle" />
    <div className="node-topline"><span>{node.kind}</span><span className={`provenance provenance-${node.provenance}`}>{node.provenance}</span></div>
    <strong>{node.label}</strong><p>{node.plainLabel}</p>
    <div className="node-footer"><span>{node.state}</span>{node.metadata?.model && <code>{node.metadata.model}</code>}</div>
    <Handle type="source" position={Position.Right} className="node-handle" />
  </motion.article>
})

function miniMapColor(node: { data?: unknown }): string {
  const tone = (node.data as LfgNodeData | undefined)?.tone
  return tone === 'amber' ? '#ffbf47' : tone === 'magenta' ? '#ff4fd8' : tone === 'cyan' ? '#42d9ff' : tone === 'red' ? '#ff5263' : '#91ff4a'
}
