"use client";

import React, { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Cell {
    x: number;
    y: number;
    isWall: boolean;
    isPath: boolean;
    isVisited: boolean;
    isStart: boolean;
    isEnd: boolean;
}

interface MazeNavigatorProps {
    width?: number;
    height?: number;
}

// Generate a simple maze
function generateMaze(width: number, height: number): Cell[][] {
    const maze: Cell[][] = [];

    for (let y = 0; y < height; y++) {
        const row: Cell[] = [];
        for (let x = 0; x < width; x++) {
            const isWall =
                x === 0 || x === width - 1 || y === 0 || y === height - 1 ||
                (Math.random() < 0.25 && x > 1 && x < width - 2 && y > 1 && y < height - 2);

            row.push({
                x,
                y,
                isWall,
                isPath: false,
                isVisited: false,
                isStart: x === 1 && y === 1,
                isEnd: x === width - 2 && y === height - 2,
            });
        }
        maze.push(row);
    }

    // Clear start and end
    maze[1][1].isWall = false;
    maze[height - 2][width - 2].isWall = false;

    return maze;
}

// BFS pathfinding
function bfs(maze: Cell[][], start: [number, number], end: [number, number]) {
    const visited: Set<string> = new Set();
    const queue: { pos: [number, number]; path: [number, number][] }[] = [];
    const visitOrder: [number, number][] = [];

    queue.push({ pos: start, path: [start] });
    visited.add(`${start[0]},${start[1]}`);

    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    while (queue.length > 0) {
        const { pos, path } = queue.shift()!;
        const [x, y] = pos;
        visitOrder.push(pos);

        if (x === end[0] && y === end[1]) {
            return { path, visitOrder };
        }

        for (const [dx, dy] of directions) {
            const nx = x + dx;
            const ny = y + dy;
            const key = `${nx},${ny}`;

            if (
                ny >= 0 && ny < maze.length &&
                nx >= 0 && nx < maze[0].length &&
                !maze[ny][nx].isWall &&
                !visited.has(key)
            ) {
                visited.add(key);
                queue.push({ pos: [nx, ny], path: [...path, [nx, ny]] });
            }
        }
    }

    return { path: [], visitOrder };
}

// DFS pathfinding
function dfs(maze: Cell[][], start: [number, number], end: [number, number]) {
    const visited: Set<string> = new Set();
    const stack: { pos: [number, number]; path: [number, number][] }[] = [];
    const visitOrder: [number, number][] = [];

    stack.push({ pos: start, path: [start] });

    const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    while (stack.length > 0) {
        const { pos, path } = stack.pop()!;
        const [x, y] = pos;
        const key = `${x},${y}`;

        if (visited.has(key)) continue;
        visited.add(key);
        visitOrder.push(pos);

        if (x === end[0] && y === end[1]) {
            return { path, visitOrder };
        }

        for (const [dx, dy] of directions) {
            const nx = x + dx;
            const ny = y + dy;

            if (
                ny >= 0 && ny < maze.length &&
                nx >= 0 && nx < maze[0].length &&
                !maze[ny][nx].isWall &&
                !visited.has(`${nx},${ny}`)
            ) {
                stack.push({ pos: [nx, ny], path: [...path, [nx, ny]] });
            }
        }
    }

    return { path: [], visitOrder };
}

export function MazeNavigator({ width = 15, height = 15 }: MazeNavigatorProps) {
    const [maze, setMaze] = useState<Cell[][]>(() => generateMaze(width, height));
    const [algorithm, setAlgorithm] = useState<"bfs" | "dfs">("bfs");
    const [isRunning, setIsRunning] = useState(false);
    const [stats, setStats] = useState<{ pathLength: number; visited: number } | null>(null);

    const regenerateMaze = useCallback(() => {
        setMaze(generateMaze(width, height));
        setStats(null);
    }, [width, height]);

    const runPathfinding = useCallback(async () => {
        setIsRunning(true);
        setStats(null);

        // Reset maze state
        const newMaze = maze.map(row =>
            row.map(cell => ({ ...cell, isPath: false, isVisited: false }))
        );
        setMaze(newMaze);

        const start: [number, number] = [1, 1];
        const end: [number, number] = [width - 2, height - 2];

        const { path, visitOrder } = algorithm === "bfs"
            ? bfs(newMaze, start, end)
            : dfs(newMaze, start, end);

        // Animate visited cells
        for (let i = 0; i < visitOrder.length; i++) {
            const [x, y] = visitOrder[i];
            setMaze(prev => {
                const updated = prev.map(row => row.map(cell => ({ ...cell })));
                updated[y][x].isVisited = true;
                return updated;
            });
            await new Promise(r => setTimeout(r, 20));
        }

        // Animate path
        for (let i = 0; i < path.length; i++) {
            const [x, y] = path[i];
            setMaze(prev => {
                const updated = prev.map(row => row.map(cell => ({ ...cell })));
                updated[y][x].isPath = true;
                return updated;
            });
            await new Promise(r => setTimeout(r, 30));
        }

        setStats({ pathLength: path.length, visited: visitOrder.length });
        setIsRunning(false);
    }, [maze, algorithm, width, height]);

    return (
        <div className="flex flex-col items-center gap-6 p-6">
            {/* Header */}
            <div className="text-center">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-green-400 to-cyan-500 bg-clip-text text-transparent">
                    Maze Navigator
                </h2>
                <p className="text-gray-400 mt-2">
                    Visualize search algorithms in action
                </p>
            </div>

            {/* Algorithm Selection */}
            <div className="flex gap-4 p-2 bg-gray-800 rounded-lg">
                {(["bfs", "dfs"] as const).map((algo) => (
                    <motion.button
                        key={algo}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setAlgorithm(algo)}
                        className={`
              px-6 py-2 rounded-md font-medium transition-all
              ${algorithm === algo
                                ? "bg-gradient-to-r from-green-500 to-cyan-500 text-white"
                                : "bg-gray-700 text-gray-400 hover:text-white"
                            }
            `}
                    >
                        {algo.toUpperCase()}
                    </motion.button>
                ))}
            </div>

            {/* Maze Grid */}
            <motion.div
                className="p-4 bg-gray-900 rounded-xl shadow-2xl border border-gray-700"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
            >
                <div
                    className="grid gap-0"
                    style={{ gridTemplateColumns: `repeat(${width}, 1fr)` }}
                >
                    {maze.flat().map((cell, idx) => (
                        <motion.div
                            key={`${cell.x}-${cell.y}`}
                            initial={cell.isVisited || cell.isPath ? { scale: 0.5 } : false}
                            animate={{ scale: 1 }}
                            className={`
                w-6 h-6 border border-gray-800
                ${cell.isWall ? "bg-gray-700" : "bg-gray-900"}
                ${cell.isStart ? "bg-blue-500" : ""}
                ${cell.isEnd ? "bg-red-500" : ""}
                ${cell.isPath && !cell.isStart && !cell.isEnd ? "bg-green-500" : ""}
                ${cell.isVisited && !cell.isPath && !cell.isStart && !cell.isEnd ? "bg-yellow-500/30" : ""}
                transition-colors duration-100
              `}
                        />
                    ))}
                </div>
            </motion.div>

            {/* Legend */}
            <div className="flex gap-6 text-sm text-gray-400">
                <span className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-blue-500 rounded" /> Start
                </span>
                <span className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-red-500 rounded" /> End
                </span>
                <span className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-yellow-500/50 rounded" /> Visited
                </span>
                <span className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-green-500 rounded" /> Path
                </span>
            </div>

            {/* Controls */}
            <div className="flex gap-4">
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={runPathfinding}
                    disabled={isRunning}
                    className={`
            px-8 py-3 rounded-lg font-bold text-lg
            ${isRunning
                            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                            : "bg-gradient-to-r from-green-500 to-cyan-500 text-white hover:from-green-600 hover:to-cyan-600"
                        }
            transition-all shadow-lg
          `}
                >
                    {isRunning ? "Searching..." : "🔍 Find Path"}
                </motion.button>

                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={regenerateMaze}
                    disabled={isRunning}
                    className="px-6 py-3 rounded-lg font-medium bg-gray-700 text-gray-300 hover:bg-gray-600 transition-all"
                >
                    New Maze
                </motion.button>
            </div>

            {/* Stats */}
            <AnimatePresence>
                {stats && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="flex gap-8 p-4 bg-gray-800 rounded-lg"
                    >
                        <div className="text-center">
                            <div className="text-2xl font-bold text-green-400">{stats.pathLength}</div>
                            <div className="text-sm text-gray-400">Path Length</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-yellow-400">{stats.visited}</div>
                            <div className="text-sm text-gray-400">Cells Visited</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-cyan-400">
                                {((stats.pathLength / stats.visited) * 100).toFixed(0)}%
                            </div>
                            <div className="text-sm text-gray-400">Efficiency</div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

export default MazeNavigator;
