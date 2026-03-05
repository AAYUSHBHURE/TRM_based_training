"use client";

import React, { useCallback, useMemo } from "react";
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    Node,
    Edge,
    Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { motion } from "framer-motion";

interface ZTraceGraphProps {
    cycle: number;
    zTraces: number[][][];
    onNodeClick?: (nodeId: string) => void;
}

// Custom node for z-states
function ZStateNode({ data }: { data: { label: string; value: number; cycle: number } }) {
    const colorScale = [
        "bg-red-500",
        "bg-orange-500",
        "bg-yellow-500",
        "bg-lime-500",
        "bg-green-500",
    ];

    const confidence = Math.min(data.value / 9, 1);
    const colorIdx = Math.floor(confidence * (colorScale.length - 1));

    return (
        <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", damping: 15 }}
            className={`
        px-4 py-2 rounded-xl shadow-lg border-2
        ${colorScale[colorIdx]} bg-opacity-80
        text-white font-bold
        hover:scale-110 transition-transform cursor-pointer
      `}
        >
            <div className="text-xs opacity-70">Cycle {data.cycle}</div>
            <div className="text-2xl">{data.value}</div>
            <div className="text-xs">{data.label}</div>
        </motion.div>
    );
}

// Custom node for input (x)
function InputNode({ data }: { data: { label: string; value: number } }) {
    return (
        <div className="px-3 py-2 rounded-lg bg-blue-600 text-white shadow-md">
            <div className="text-xs opacity-70">Input</div>
            <div className="text-xl font-bold">{data.value || "?"}</div>
        </div>
    );
}

// Custom node for output (y_hat)
function OutputNode({ data }: { data: { label: string; value: number; confidence: number } }) {
    return (
        <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="px-4 py-3 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-xl"
        >
            <div className="text-xs opacity-70">Output</div>
            <div className="text-3xl font-bold">{data.value}</div>
            <div className="text-xs">{(data.confidence * 100).toFixed(0)}% confident</div>
        </motion.div>
    );
}

const nodeTypes = {
    zState: ZStateNode,
    input: InputNode,
    output: OutputNode,
};

export function ZTraceGraph({ cycle, zTraces, onNodeClick }: ZTraceGraphProps) {
    // Generate nodes from z-traces
    const generateNodes = useCallback((): Node[] => {
        const nodes: Node[] = [];
        const spacing = 150;

        // Input node (center)
        nodes.push({
            id: "input-0",
            type: "input",
            position: { x: 0, y: 0 },
            data: { label: "x", value: 5 },
        });

        // Z-trace nodes in a spiral/tree pattern
        zTraces.forEach((cycleTraces, cycleIdx) => {
            if (cycleIdx >= cycle) return;

            cycleTraces.forEach((zStep, stepIdx) => {
                const angle = (stepIdx / cycleTraces.length) * Math.PI * 2;
                const radius = (cycleIdx + 1) * spacing;

                nodes.push({
                    id: `z-${cycleIdx}-${stepIdx}`,
                    type: "zState",
                    position: {
                        x: Math.cos(angle) * radius,
                        y: Math.sin(angle) * radius + (cycleIdx + 1) * 100,
                    },
                    data: {
                        label: `z[${stepIdx}]`,
                        value: zStep[0] || Math.floor(Math.random() * 9) + 1,
                        cycle: cycleIdx + 1,
                    },
                });
            });
        });

        // Output node
        if (cycle >= zTraces.length && zTraces.length > 0) {
            nodes.push({
                id: "output-0",
                type: "output",
                position: { x: 0, y: (zTraces.length + 1) * 150 },
                data: { label: "ŷ", value: 7, confidence: 0.92 },
            });
        }

        return nodes;
    }, [cycle, zTraces]);

    // Generate edges connecting the nodes
    const generateEdges = useCallback((): Edge[] => {
        const edges: Edge[] = [];

        zTraces.forEach((cycleTraces, cycleIdx) => {
            if (cycleIdx >= cycle) return;

            cycleTraces.forEach((_, stepIdx) => {
                // Connect to previous
                if (stepIdx > 0) {
                    edges.push({
                        id: `e-${cycleIdx}-${stepIdx - 1}-${stepIdx}`,
                        source: `z-${cycleIdx}-${stepIdx - 1}`,
                        target: `z-${cycleIdx}-${stepIdx}`,
                        animated: true,
                        style: { stroke: "#888" },
                    });
                } else if (cycleIdx === 0) {
                    // Connect first z to input
                    edges.push({
                        id: `e-input-${cycleIdx}-${stepIdx}`,
                        source: "input-0",
                        target: `z-${cycleIdx}-${stepIdx}`,
                        animated: true,
                        style: { stroke: "#3b82f6" },
                    });
                } else {
                    // Connect to previous cycle's last
                    const prevLastIdx = zTraces[cycleIdx - 1].length - 1;
                    edges.push({
                        id: `e-cross-${cycleIdx - 1}-${cycleIdx}`,
                        source: `z-${cycleIdx - 1}-${prevLastIdx}`,
                        target: `z-${cycleIdx}-${stepIdx}`,
                        animated: true,
                        style: { stroke: "#10b981" },
                    });
                }
            });
        });

        // Connect last z to output
        if (cycle >= zTraces.length && zTraces.length > 0) {
            const lastCycle = zTraces.length - 1;
            const lastStep = zTraces[lastCycle].length - 1;
            edges.push({
                id: "e-output",
                source: `z-${lastCycle}-${lastStep}`,
                target: "output-0",
                animated: true,
                style: { stroke: "#a855f7", strokeWidth: 3 },
            });
        }

        return edges;
    }, [cycle, zTraces]);

    const nodes = useMemo(() => generateNodes(), [generateNodes]);
    const edges = useMemo(() => generateEdges(), [generateEdges]);

    const [displayNodes, setNodes, onNodesChange] = useNodesState(nodes);
    const [displayEdges, setEdges, onEdgesChange] = useEdgesState(edges);

    // Update when traces change
    React.useEffect(() => {
        setNodes(nodes);
        setEdges(edges);
    }, [nodes, edges, setNodes, setEdges]);

    return (
        <div className="w-full h-[600px] rounded-xl overflow-hidden border border-gray-700 bg-gray-900">
            <ReactFlow
                nodes={displayNodes}
                edges={displayEdges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                fitView
                minZoom={0.5}
                maxZoom={2}
                defaultEdgeOptions={{ animated: true }}
            >
                <Background color="#374151" gap={20} />
                <Controls className="bg-gray-800 rounded-lg" />
                <MiniMap
                    nodeColor={(n) => {
                        if (n.type === "input") return "#3b82f6";
                        if (n.type === "output") return "#a855f7";
                        return "#10b981";
                    }}
                    className="bg-gray-800 rounded-lg"
                />
            </ReactFlow>
        </div>
    );
}

export default ZTraceGraph;
