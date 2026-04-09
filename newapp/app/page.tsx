"use client"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ArrowLeft, Home, Columns2, Square, ChevronDown } from "lucide-react"

// ============ CSSLab Logo ============
function CSSLabLogo({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <svg className="h-10 w-10" viewBox="-10 -10 120 120" aria-hidden="true">
        <g className="outer-edges">
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 50 0 A 50 50 0 0 1 79.39 7.73"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 79.39 7.73 A 50 50 0 0 1 97.55 29.39"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 97.55 29.39 A 50 50 0 0 1 97.55 70.61"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 79.39 92.27 A 50 50 0 0 1 50 100"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 50 100 A 50 50 0 0 1 20.61 92.27"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 20.61 92.27 A 50 50 0 0 1 2.45 70.61"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 2.45 70.61 A 50 50 0 0 1 2.45 29.39"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 2.45 29.39 A 50 50 0 0 1 20.61 7.73"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="4" strokeLinecap="round" d="M 20.61 7.73 A 50 50 0 0 1 50 0"/>
        </g>
        <g className="inner-edges">
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 50 0 Q 59.5 39 97.55 29.39"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 50 0 Q 40.5 39 2.45 29.39"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 79.39 7.73 Q 70 50 97.55 70.61"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 79.39 7.73 Q 50 15 20.61 7.73"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 97.55 29.39 Q 79 65 79.39 92.27"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 97.55 70.61 Q 70 85 50 100"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 79.39 92.27 Q 50 100 20.61 92.27"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 50 100 Q 30 85 2.45 70.61"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 20.61 92.27 Q 21 65 2.45 29.39"/>
          <path fill="none" stroke="#4a90d9" strokeWidth="3" strokeLinecap="round" d="M 2.45 70.61 Q 30 50 20.61 7.73"/>
        </g>
        <g className="shortcuts">
          <path fill="none" stroke="#6bb3f0" strokeWidth="4" strokeLinecap="round" d="M 97.55 70.61 Q 50 50 2.45 29.39"/>
          <path fill="none" stroke="#6bb3f0" strokeWidth="4" strokeLinecap="round" d="M 79.39 92.27 Q 50 50 20.61 7.73"/>
        </g>
        <g className="nodes">
          <circle fill="#7ec8f3" cx="50" cy="0" r="5.5"/>
          <circle fill="#7ec8f3" cx="79.39" cy="7.73" r="5.5"/>
          <circle fill="#7ec8f3" cx="97.55" cy="29.39" r="5.5"/>
          <circle fill="#7ec8f3" cx="97.55" cy="70.61" r="5.5"/>
          <circle fill="#7ec8f3" cx="79.39" cy="92.27" r="5.5"/>
          <circle fill="#7ec8f3" cx="50" cy="100" r="5.5"/>
          <circle fill="#7ec8f3" cx="20.61" cy="92.27" r="5.5"/>
          <circle fill="#7ec8f3" cx="2.45" cy="70.61" r="5.5"/>
          <circle fill="#7ec8f3" cx="2.45" cy="29.39" r="5.5"/>
          <circle fill="#7ec8f3" cx="20.61" cy="7.73" r="5.5"/>
        </g>
      </svg>
      <div className="text-2xl tracking-wide">
        <span className="font-bold">CSS</span><span className="font-normal">Lab</span>
      </div>
    </div>
  )
}

// ============ Navigation Buttons ============
function NavButtons({ onHome, onBack }: { onHome: () => void; onBack?: () => void }) {
  return (
    <div className="flex items-center gap-2">
      <Button variant="ghost" size="icon" onClick={onHome} title="Home">
        <Home className="h-5 w-5" />
      </Button>
      {onBack && (
        <Button variant="ghost" size="icon" onClick={onBack} title="Back">
          <ArrowLeft className="h-5 w-5" />
        </Button>
      )}
    </div>
  )
}

// ============ Floor Plan ============
const rooms = [
  { x: 20, y: 20, width: 60, height: 50 },
  { x: 20, y: 90, width: 80, height: 80 },
  { x: 20, y: 190, width: 60, height: 50 },
  { x: 20, y: 260, width: 40, height: 40 },
  { x: 180, y: 20, width: 180, height: 60 },
  { x: 180, y: 100, width: 180, height: 160 },
  { x: 180, y: 280, width: 80, height: 40 },
  { x: 280, y: 280, width: 80, height: 40 },
  { x: 420, y: 20, width: 140, height: 50 },
  { x: 420, y: 90, width: 140, height: 100 },
  { x: 420, y: 210, width: 140, height: 110 },
]

const pointColors = { orange: "#F5A623", pink: "#E57373", purple: "#7B68EE" }

function FloorPlan({ dataPoints, selectedTimeRange, compact = false }: { dataPoints: DataPoint[]; selectedTimeRange: string | null; compact?: boolean }) {
  const filtered = selectedTimeRange ? dataPoints.filter((p) => p.time === selectedTimeRange) : dataPoints
  return (
    <div className="relative w-full">
      <svg viewBox="0 0 600 340" className={`w-full ${compact ? "h-32" : "h-auto"}`} style={{ backgroundColor: "#d4d4d4" }} preserveAspectRatio={compact ? "xMidYMid slice" : "xMidYMid meet"}>
        {rooms.map((room, i) => (
          <rect key={i} x={room.x} y={room.y} width={room.width} height={room.height} fill="#888" stroke="#777" strokeWidth="1" />
        ))}
        {filtered.map((p) => (
          <g key={p.id}>
            <circle cx={p.x} cy={p.y} r={12} fill="none" stroke={pointColors[p.color]} strokeWidth="2" />
            <circle cx={p.x} cy={p.y} r={4} fill={pointColors[p.color]} />
          </g>
        ))}
      </svg>
    </div>
  )
}

// ============ Timeline ============
const segmentColors = { orange: "bg-[#F5A623]", pink: "bg-[#E8B4B8]", purple: "bg-[#9B8BB8]" }

function Timeline({ segments, selectedSegment, onSegmentClick }: { segments: TimeSegment[]; selectedSegment: string | null; onSegmentClick: (id: string | null) => void }) {
  return (
    <div className="w-full">
      <p className="text-center text-muted-foreground mb-2 text-sm">Timestamps</p>
      <div className="relative">
        <div className="relative h-10 bg-card border border-border rounded-sm overflow-hidden">
          {segments.map((seg) => (
            <button
              key={seg.id}
              onClick={() => onSegmentClick(selectedSegment === seg.id ? null : seg.id)}
              className={`absolute top-0 h-full transition-all ${segmentColors[seg.color]} ${selectedSegment === seg.id ? "ring-2 ring-foreground/50 ring-offset-1" : ""} ${selectedSegment && selectedSegment !== seg.id ? "opacity-40" : ""}`}
              style={{ left: `${seg.startPercent}%`, width: `${seg.widthPercent}%` }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-sm text-muted-foreground px-1">
          <span>08 AM</span>
          <span>09 AM</span>
          <span>10 AM</span>
        </div>
      </div>
    </div>
  )
}

// ============ Placeholder Chart ============
function AccuracyChart({ algorithmName }: { algorithmName: string }) {
  return (
    <div className="w-full h-full bg-card border border-border rounded-lg p-4 flex flex-col">
      <p className="text-base font-semibold text-center mb-2">{algorithmName} accuracy in 2-stop trajectory</p>
      <div className="flex-1 relative">
        <svg viewBox="0 0 300 200" className="w-full h-full">
          {/* Y-axis */}
          <line x1="40" y1="20" x2="40" y2="170" stroke="#888" strokeWidth="1"/>
          {/* X-axis */}
          <line x1="40" y1="170" x2="280" y2="170" stroke="#888" strokeWidth="1"/>
          {/* Y-axis labels */}
          <text x="35" y="25" fontSize="8" textAnchor="end" fill="#666">1.0</text>
          <text x="35" y="65" fontSize="8" textAnchor="end" fill="#666">0.8</text>
          <text x="35" y="105" fontSize="8" textAnchor="end" fill="#666">0.6</text>
          <text x="35" y="145" fontSize="8" textAnchor="end" fill="#666">0.4</text>
          <text x="35" y="175" fontSize="8" textAnchor="end" fill="#666">0.2</text>
          {/* X-axis labels */}
          <text x="60" y="185" fontSize="8" textAnchor="middle" fill="#666">20</text>
          <text x="120" y="185" fontSize="8" textAnchor="middle" fill="#666">60</text>
          <text x="180" y="185" fontSize="8" textAnchor="middle" fill="#666">100</text>
          <text x="240" y="185" fontSize="8" textAnchor="middle" fill="#666">140</text>
          {/* Axis labels */}
          <text x="10" y="100" fontSize="8" fill="#666" transform="rotate(-90, 10, 100)">Accuracy</text>
          <text x="160" y="198" fontSize="8" textAnchor="middle" fill="#666">Roaming Distance (m)</text>
          {/* Scatter points */}
          {[...Array(50)].map((_, i) => (
            <circle key={i} cx={50 + Math.random() * 220} cy={30 + Math.random() * 120} r="2" fill="#6b9fd4" opacity="0.5"/>
          ))}
          {/* Mean line */}
          <path d="M 50 40 Q 100 35 150 50 T 250 80" fill="none" stroke="#2563eb" strokeWidth="2"/>
          {/* Dashed reference line */}
          <line x1="50" y1="45" x2="270" y2="45" stroke="#666" strokeWidth="1" strokeDasharray="4,2"/>
        </svg>
      </div>
      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-sm mt-3 justify-center">
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-600"></span>Mean</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 text-red-500">x</span>User maxima</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 border-t border-dashed border-gray-600"></span>Max mean</span>
      </div>
    </div>
  )
}

// ============ TYPES & DATA ============

type DataPoint = { id: string; x: number; y: number; time: string; color: "orange" | "pink" | "purple" }
type TimeSegment = { id: string; startPercent: number; widthPercent: number; color: "orange" | "pink" | "purple" }

const algorithms = [
  { 
    id: "lachesis", 
    name: "Lachesis", 
    shortDesc: "Roaming-distance based stop detection",
    description: "Lachesis is a greedy, sequential and threshold-based algorithm taking in three parameters: dur_min, dt_max, and delta_roam. The algorithm processes pings in time order, and builds a candidate stop by scanning trajectories and grouping consecutive pings into a cluster under the following conditions: (1) the cluster diameter is less than or equal to delta_roam, (2) has no temporal gaps larger than dt_max, and (3) the accumulated duration is equal to or greater than dur_min. Key limitations include (1) being highly sensitive to parameter choices, with a small delta_roam leading to splitting or missing stops, and a large delta_roam leading to merging stops, and (2) assuming continuous sampling, so GPS noise or sparse data can lead to fragmented stops.", 
    source: "Lachesis: A GPS-Based Human Mobility Pattern Analyzer",
    parameters: [
      { name: "dur_min", defaultValue: "60", description: "Minimum duration for a stop (seconds)" }, 
      { name: "dt_max", defaultValue: "60", description: "Maximum allowed temporal gap between pings (seconds)" },
      { name: "delta_roam", defaultValue: "25", description: "Maximum roaming distance (meters)" }
    ] 
  },
  { 
    id: "seqscan", 
    name: "SeqScan", 
    shortDesc: "Sequential scanning with thresholds",
    description: "SeqScan extends DBSCAN to spatio-temporal data while guaranteeing temporal separation between detected stops. Each stop is defined as a DBSCAN cluster computed over a local time context, using the reachability graph of core points (a ping with a minimum of min_pts neighbors at a distance below dist_thresh). The algorithm scans points in chronological order, maintaining both an active cluster and an active time context for neighbor queries. For each incoming ping, neighbors are queried within the active time context; if it is connected to the active cluster, the active cluster is expanded. Otherwise, DBSCAN is applied to the tail of the trajectory, starting from the last ping of the active cluster up to the incoming ping. If a cluster with duration at least dur_min is found in the tail, it becomes the new active cluster and the active time context is shifted, continuing until reaching the end of the trajectory.", 
    source: "Sequential Scan Algorithm for GPS Stop Detection",
    parameters: [
      { name: "time_thresh", defaultValue: "300", description: "Time threshold for stop detection" }, 
      { name: "dist_thresh", defaultValue: "50", description: "Distance threshold (meters)" }
    ] 
  },
  { 
    id: "dbscan", 
    name: "ST-DBSCAN", 
    shortDesc: "Density-based spatiotemporal clustering",
    description: "ST-DBSCAN extends classical DBSCAN by incorporating a second scale parameter to avoid long temporal gaps within detected stops, namely T_max, in addition to the spatial scale parameter eps_spatial, and minPts, the usual density threshold in DBSCAN. In this way, it incorporates both spatial and temporal proximity to define reachability. Since stops are defined as components of the reachability graph of core points, ST-DBSCAN is robust to noise points that could break up stops in sequential algorithms. Standard ST-DBSCAN does not guarantee temporal separation, so clusters can overlap in time even if they represent different stops. In this dashboard, we extend it with a simple heuristic to remove overlaps, where each core point can act as a cutoff and prevent merging non-neighbor core points.", 
    source: "ST-DBSCAN: An algorithm for clustering spatial-temporal data",
    parameters: [
      { name: "eps_spatial", defaultValue: "50", description: "Spatial radius epsilon (meters)" }, 
      { name: "T_max", defaultValue: "300", description: "Maximum temporal gap for neighborhood linkage (seconds)" }, 
      { name: "minPts", defaultValue: "3", description: "Minimum neighbors required for a core point" }
    ] 
  },
  { 
    id: "gridbased", 
    name: "Grid-based", 
    shortDesc: "Cell-based dwell time analysis",
    description: "Divides the geographic area into grid cells and identifies stops based on dwell time within cells, useful for aggregated analysis at different spatial resolutions.", 
    source: "Grid-based Methods for Mobility Analysis",
    parameters: [
      { name: "cell_size", defaultValue: "100", description: "Grid cell size (meters)" }, 
      { name: "min_time", defaultValue: "300", description: "Minimum time in cell (seconds)" }
    ] 
  },
  { 
    id: "medoid", 
    name: "Medoid-based", 
    shortDesc: "Representative point clustering",
    description: "Uses medoid computation to find representative central points of potential stops, robust against outliers in noisy GPS data.", 
    source: "Medoid-based Clustering for GPS Trajectories",
    parameters: [
      { name: "radius", defaultValue: "50", description: "Search radius (meters)" }, 
      { name: "min_duration", defaultValue: "60", description: "Minimum stop duration" }
    ] 
  },
  { 
    id: "incremental", 
    name: "Incremental", 
    shortDesc: "Online clustering approach",
    description: "Online algorithm that processes GPS points as they arrive, building clusters incrementally without needing the full dataset upfront. Ideal for streaming data.", 
    source: "Incremental Clustering for Mobile Sensor Data",
    parameters: [
      { name: "threshold", defaultValue: "50", description: "Distance threshold (meters)" }, 
      { name: "min_pts", defaultValue: "3", description: "Minimum points per cluster" }
    ] 
  },
]

const sampleDataPoints: DataPoint[] = [
  { id: "1", x: 50, y: 110, time: "orange", color: "orange" }, { id: "2", x: 65, y: 100, time: "orange", color: "orange" },
  { id: "3", x: 80, y: 115, time: "orange", color: "orange" }, { id: "4", x: 55, y: 140, time: "orange", color: "orange" },
  { id: "5", x: 260, y: 130, time: "pink", color: "pink" }, { id: "6", x: 280, y: 145, time: "pink", color: "pink" },
  { id: "7", x: 295, y: 155, time: "pink", color: "pink" }, { id: "8", x: 275, y: 170, time: "pink", color: "pink" },
  { id: "9", x: 290, y: 185, time: "pink", color: "pink" }, { id: "10", x: 305, y: 160, time: "pink", color: "pink" },
  { id: "11", x: 265, y: 200, time: "pink", color: "pink" }, { id: "12", x: 290, y: 295, time: "pink", color: "pink" },
  { id: "13", x: 475, y: 100, time: "purple", color: "purple" }, { id: "14", x: 490, y: 135, time: "purple", color: "purple" },
  { id: "15", x: 505, y: 150, time: "purple", color: "purple" }, { id: "16", x: 520, y: 115, time: "purple", color: "purple" },
  { id: "17", x: 540, y: 130, time: "purple", color: "purple" }, { id: "18", x: 475, y: 250, time: "purple", color: "purple" },
  { id: "19", x: 495, y: 270, time: "purple", color: "purple" }, { id: "20", x: 530, y: 255, time: "purple", color: "purple" },
]

const timelineSegments: TimeSegment[] = [
  { id: "orange", startPercent: 10, widthPercent: 8, color: "orange" },
  { id: "pink", startPercent: 30, widthPercent: 25, color: "pink" },
  { id: "purple", startPercent: 68, widthPercent: 20, color: "purple" },
]

const generateDataPoints = (): DataPoint[] => {
  const colors = ["orange", "pink", "purple"] as const
  const clusters = [{ cx: 60, cy: 120, count: 4, color: colors[0] }, { cx: 280, cy: 160, count: 8, color: colors[1] }, { cx: 500, cy: 180, count: 6, color: colors[2] }]
  const points: DataPoint[] = []
  let id = 1
  for (const c of clusters) {
    for (let i = 0; i < c.count; i++) {
      points.push({ id: String(id++), x: c.cx + (Math.random() - 0.5) * 60, y: c.cy + (Math.random() - 0.5) * 60, time: c.color, color: c.color })
    }
  }
  return points
}

// ============ MAIN COMPONENT ============

export default function NomadDashboard() {
  const [activeView, setActiveView] = useState<"home" | "learn" | "demo" | "metrics">("home")
  const [previousView, setPreviousView] = useState<"home" | "learn" | "demo" | "metrics">("home")
  const [selectedTimeRange, setSelectedTimeRange] = useState<string | null>(null)
  const [expandedAlgorithm, setExpandedAlgorithm] = useState<string | null>(null)
  const [compareMode, setCompareMode] = useState(false)
  const [leftAlgorithm, setLeftAlgorithm] = useState("lachesis")
  const [rightAlgorithm, setRightAlgorithm] = useState("dbscan")
  const [leftParams, setLeftParams] = useState<Record<string, string>>({})
  const [rightParams, setRightParams] = useState<Record<string, string>>({})
  const [leftData, setLeftData] = useState(generateDataPoints)
  const [rightData, setRightData] = useState(generateDataPoints)

  const handleVisualize = useCallback((side: "left" | "right") => {
    if (side === "left") setLeftData(generateDataPoints())
    else setRightData(generateDataPoints())
  }, [])

  const handleNewTrajectory = useCallback(() => {
    setLeftData(generateDataPoints())
    setRightData(generateDataPoints())
  }, [])

  const handleAlgorithmSelect = (algorithmId: string) => {
    setLeftAlgorithm(algorithmId)
    setExpandedAlgorithm(null)
    setActiveView("demo")
  }

  const goHome = () => {
    setActiveView("home")
    setExpandedAlgorithm(null)
  }

  const goBack = () => {
    if (activeView === "metrics") {
      setActiveView("demo")
    } else {
      setActiveView(previousView)
    }
    setExpandedAlgorithm(null)
  }

  const navigateTo = (view: "home" | "learn" | "demo" | "metrics") => {
    setPreviousView(activeView)
    setActiveView(view)
  }

  const leftAlgo = algorithms.find((a) => a.id === leftAlgorithm)
  const rightAlgo = algorithms.find((a) => a.id === rightAlgorithm)
  return (
    <div className="h-screen overflow-hidden bg-background flex flex-col">
      {/* HOME VIEW */}
      {activeView === "home" && (
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex items-center justify-between">
            <CSSLabLogo />
          </div>
          <div className="grid lg:grid-cols-2 gap-4 items-start overflow-hidden">
            <div className="flex flex-col gap-3 overflow-hidden">
              <h2 className="text-3xl font-bold text-foreground">Stop Detection Algorithm Visualizer</h2>
              <p className="text-muted-foreground text-base">
                Visualize and compare different stop detection algorithms to understand how they identify stops in GPS trajectory data.
              </p>
              <Card className="bg-card">
                <CardContent className="pt-5 space-y-3">
                  <h3 className="font-semibold text-foreground text-xl">What this tool does</h3>
                  <ul className="text-base text-muted-foreground space-y-1">
                    <li>Visualize different stop detection algorithms in the same floor plan context.</li>
                    <li>Compare algorithm performance side-by-side with synchronized time windows.</li>
                    <li>Inspect algorithm parameters and see how they change stop clusters.</li>
                  </ul>
                </CardContent>
              </Card>
              <div className="grid sm:grid-cols-3 gap-2 text-sm">
                <Card className="bg-muted/40">
                  <CardContent className="p-4">
                    <p className="text-muted-foreground">Algorithms</p>
                    <p className="text-3xl font-bold">{algorithms.length}</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/40">
                  <CardContent className="p-4">
                    <p className="text-muted-foreground">Sample pings</p>
                    <p className="text-3xl font-bold">{sampleDataPoints.length}</p>
                  </CardContent>
                </Card>
                <Card className="bg-muted/40">
                  <CardContent className="p-4">
                    <p className="text-muted-foreground">Timeline blocks</p>
                    <p className="text-3xl font-bold">{timelineSegments.length}</p>
                  </CardContent>
                </Card>
              </div>
              <div className="flex gap-3">
                <Button variant="outline" size="lg" onClick={() => navigateTo("learn")}>Learn Algorithms</Button>
                <Button size="lg" onClick={() => navigateTo("demo")}>Demo</Button>
              </div>
            </div>
            <div className="flex flex-col gap-2 overflow-hidden">
              <Card className="bg-card overflow-hidden">
                <CardContent className="p-0">
                  <FloorPlan dataPoints={sampleDataPoints} selectedTimeRange={selectedTimeRange} />
                </CardContent>
              </Card>
              <Timeline segments={timelineSegments} selectedSegment={selectedTimeRange} onSegmentClick={setSelectedTimeRange} />
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Cluster Legend</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-3 gap-3 text-sm">
                  <div className="rounded-md bg-muted/40 p-3">
                    <p className="text-muted-foreground">Morning home stop</p>
                    <p className="font-semibold text-[#F5A623]">Orange cluster</p>
                  </div>
                  <div className="rounded-md bg-muted/40 p-3">
                    <p className="text-muted-foreground">Mid-session pause</p>
                    <p className="font-semibold text-[#E57373]">Pink cluster</p>
                  </div>
                  <div className="rounded-md bg-muted/40 p-3">
                    <p className="text-muted-foreground">Late visit zone</p>
                    <p className="font-semibold text-[#7B68EE]">Purple cluster</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* LEARN VIEW */}
      {activeView === "learn" && !expandedAlgorithm && (
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-4">
              <NavButtons onHome={goHome} onBack={goBack} />
              <h2 className="text-3xl font-bold">Learn Algorithms</h2>
            </div>
            <CSSLabLogo />
          </div>
          <p className="text-sm text-muted-foreground">
            Select an algorithm card to inspect assumptions, parameter behavior, and a method sketch before running the demo.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 overflow-hidden">
            {algorithms.map((algo) => (
              <Card
                key={algo.id}
                className="cursor-pointer hover:shadow-lg transition-all border-2 border-border hover:border-primary/50 flex flex-col"
                onClick={() => setExpandedAlgorithm(algo.id)}
              >
                <CardHeader className="pb-2 shrink-0">
                  <CardTitle className="text-lg">{algo.name}</CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col gap-3">
                  <p className="text-sm text-muted-foreground">{algo.shortDesc}</p>
                  {/* Placeholder diagram */}
                  <div className="bg-muted/50 rounded-md flex items-center justify-center min-h-[120px]">
                    <svg viewBox="0 0 100 60" className="w-full h-full max-h-28 p-2">
                      <circle cx="20" cy="30" r="8" fill="none" stroke="#888" strokeWidth="1.5"/>
                      <circle cx="50" cy="20" r="6" fill="none" stroke="#888" strokeWidth="1.5"/>
                      <circle cx="80" cy="35" r="10" fill="none" stroke="#888" strokeWidth="1.5"/>
                      <circle cx="20" cy="30" r="2" fill="#888"/>
                      <circle cx="50" cy="20" r="2" fill="#888"/>
                      <circle cx="80" cy="35" r="2" fill="#888"/>
                      <path d="M 20 30 L 50 20 L 80 35" fill="none" stroke="#888" strokeWidth="1" strokeDasharray="2,2"/>
                    </svg>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Tap to open details, parameter definitions, and a direct path to the interactive demo.
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* EXPANDED ALGORITHM VIEW */}
      {activeView === "learn" && expandedAlgorithm && (
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-4">
              <NavButtons onHome={goHome} onBack={() => setExpandedAlgorithm(null)} />
              <h2 className="text-2xl font-bold">{algorithms.find(a => a.id === expandedAlgorithm)?.name}</h2>
            </div>
            <CSSLabLogo />
          </div>
          {(() => {
            const algo = algorithms.find(a => a.id === expandedAlgorithm)
            if (!algo) return null
            return (
              <div className="flex-1 flex flex-col bg-muted/30 rounded-xl p-4 gap-3 overflow-hidden">
                {/* Description */}
                <Card className="shrink-0">
                  <CardContent className="py-4">
                    <p className="text-sm text-muted-foreground">{algo.description}</p>
                    <p className="text-sm text-muted-foreground mt-2">Source: {algo.source}</p>
                  </CardContent>
                </Card>
                {/* Diagram and Parameters */}
                <div className="grid md:grid-cols-2 gap-4">
                  {/* Diagram placeholder */}
                  <Card>
                    <CardContent className="p-4 h-full flex items-center justify-center">
                      <svg viewBox="0 0 200 150" className="w-full h-full max-h-56">
                        <circle cx="40" cy="75" r="20" fill="none" stroke="#4a90d9" strokeWidth="2"/>
                        <circle cx="100" cy="50" r="15" fill="none" stroke="#4a90d9" strokeWidth="2"/>
                        <circle cx="160" cy="80" r="25" fill="none" stroke="#4a90d9" strokeWidth="2"/>
                        <circle cx="40" cy="75" r="4" fill="#4a90d9"/>
                        <circle cx="100" cy="50" r="4" fill="#4a90d9"/>
                        <circle cx="160" cy="80" r="4" fill="#4a90d9"/>
                        <path d="M 40 75 L 100 50 L 160 80" fill="none" stroke="#6bb3f0" strokeWidth="2" strokeDasharray="4,2"/>
                        <text x="100" y="130" textAnchor="middle" fontSize="10" fill="#666">Algorithm Diagram</text>
                      </svg>
                    </CardContent>
                  </Card>
                  {/* Parameters table */}
                  <Card className="overflow-hidden">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Parameters</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="rounded-md border overflow-hidden">
                        <table className="w-full text-sm">
                          <thead className="bg-muted">
                            <tr>
                              <th className="px-3 py-2 text-left font-medium">Name</th>
                              <th className="px-3 py-2 text-left font-medium">Description</th>
                            </tr>
                          </thead>
                          <tbody>
                            {algo.parameters.map((param, idx) => (
                              <tr key={param.name} className={idx % 2 === 0 ? "bg-muted/30" : ""}>
                                <td className="px-3 py-2 font-mono text-sm">{param.name}</td>
                                <td className="px-3 py-2 text-muted-foreground text-sm">{param.description}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                {/* Demo button */}
                <Button onClick={() => handleAlgorithmSelect(algo.id)} size="lg" className="shrink-0">
                  Demo
                </Button>
              </div>
            )
          })()}
        </div>
      )}

      {/* DEMO VIEW */}
      {activeView === "demo" && (
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-4">
              <NavButtons onHome={goHome} onBack={goBack} />
              <h2 className="text-2xl font-bold">{compareMode ? "Compare Algorithms" : "Demo"}</h2>
            </div>
            <div className="flex items-center gap-3">
              <Button variant={compareMode ? "default" : "outline"} size="lg" onClick={() => setCompareMode(!compareMode)} className="gap-2">
                {compareMode ? <><Square className="h-4 w-4" /> Single</> : <><Columns2 className="h-4 w-4" /> Compare</>}
              </Button>
              <CSSLabLogo />
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Tune parameters and inspect how each algorithm reshapes stop clusters over the same spatial trajectory.
          </p>

          {!compareMode ? (
            <div className="grid lg:grid-cols-[1fr_330px] gap-3 items-start overflow-hidden">
              <div className="flex flex-col gap-3">
                <Card className="overflow-hidden">
                  <CardHeader className="py-2 px-3 bg-muted/30 shrink-0">
                    <CardTitle className="text-base font-semibold">{leftAlgo?.name}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <FloorPlan dataPoints={leftData} selectedTimeRange={selectedTimeRange} />
                  </CardContent>
                </Card>
                <Timeline segments={timelineSegments} selectedSegment={selectedTimeRange} onSegmentClick={setSelectedTimeRange} />
              </div>
              <Card className="overflow-hidden">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg font-semibold">{leftAlgo?.name || "Select Algorithm"}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Select value={leftAlgorithm} onValueChange={setLeftAlgorithm}>
                    <SelectTrigger className="h-10 text-base"><SelectValue /></SelectTrigger>
                    <SelectContent>{algorithms.map((a) => <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>)}</SelectContent>
                  </Select>
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Parameters</h4>
                    {leftAlgo?.parameters.map((p) => (
                      <div key={p.name} className="flex items-center gap-2">
                        <label className="text-sm font-mono w-24 shrink-0">{p.name}:</label>
                        <Input value={leftParams[p.name] || p.defaultValue} onChange={(e) => setLeftParams((prev) => ({ ...prev, [p.name]: e.target.value }))} className="h-9 text-sm" />
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2 pt-2">
                    <Button variant="outline" onClick={handleNewTrajectory} size="lg" className="w-full">New Trajectory</Button>
                    <Button onClick={() => handleVisualize("left")} size="lg" className="w-full">Visualize</Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <div className="flex flex-col gap-2 overflow-hidden">
              <div className="grid lg:grid-cols-2 gap-3 overflow-hidden">
                {/* Left Algorithm */}
                <div className="flex flex-col gap-2">
                  <Card className="overflow-hidden">
                    <CardHeader className="py-2 px-3 bg-muted/30 shrink-0">
                      <CardTitle className="text-base font-semibold">{leftAlgo?.name}</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      <FloorPlan dataPoints={leftData} selectedTimeRange={selectedTimeRange} />
                    </CardContent>
                  </Card>
                  <div className="flex gap-2 items-center shrink-0">
                    <Select value={leftAlgorithm} onValueChange={setLeftAlgorithm}>
                      <SelectTrigger className="h-10 text-base flex-1"><SelectValue /></SelectTrigger>
                      <SelectContent>{algorithms.map((a) => <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>)}</SelectContent>
                    </Select>
                    <Card className="px-3 py-2 text-sm">
                      {leftAlgo?.parameters.map((p) => (
                        <div key={p.name}>{p.name} = {leftParams[p.name] || p.defaultValue}</div>
                      ))}
                    </Card>
                  </div>
                </div>
                {/* Right Algorithm */}
                <div className="flex flex-col gap-2">
                  <Card className="overflow-hidden">
                    <CardHeader className="py-2 px-3 bg-muted/30 shrink-0">
                      <CardTitle className="text-base font-semibold">{rightAlgo?.name}</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      <FloorPlan dataPoints={rightData} selectedTimeRange={selectedTimeRange} />
                    </CardContent>
                  </Card>
                  <div className="flex gap-2 items-center shrink-0">
                    <Select value={rightAlgorithm} onValueChange={setRightAlgorithm}>
                      <SelectTrigger className="h-10 text-base flex-1"><SelectValue /></SelectTrigger>
                      <SelectContent>{algorithms.map((a) => <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>)}</SelectContent>
                    </Select>
                    <Card className="px-3 py-2 text-sm">
                      {rightAlgo?.parameters.map((p) => (
                        <div key={p.name}>{p.name} = {rightParams[p.name] || p.defaultValue}</div>
                      ))}
                    </Card>
                  </div>
                </div>
              </div>
              <div className="flex gap-3 shrink-0">
                <Button variant="outline" onClick={handleNewTrajectory} size="lg">New Trajectory</Button>
                <Button onClick={() => navigateTo("metrics")} size="lg" className="flex-1">Aggregated Metrics</Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* AGGREGATED METRICS VIEW */}
      {activeView === "metrics" && (
        <div className="flex-1 flex flex-col p-4 gap-3 overflow-hidden">
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-4">
              <NavButtons onHome={goHome} onBack={goBack} />
              <h2 className="text-2xl font-bold">Aggregated Metrics</h2>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="outline" size="lg" onClick={handleNewTrajectory}>New Trajectory</Button>
              <CSSLabLogo />
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Compare algorithm reliability curves and inspect parameter context for each method in the same experiment run.
          </p>

          {/* Algorithm selectors with params */}
          <div className="grid lg:grid-cols-2 gap-4 shrink-0">
            <div className="flex items-center gap-3">
              <Select value={leftAlgorithm} onValueChange={setLeftAlgorithm}>
                <SelectTrigger className="h-10 w-48 text-base">
                  <SelectValue />
                  <ChevronDown className="h-4 w-4 ml-2" />
                </SelectTrigger>
                <SelectContent>{algorithms.map((a) => <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>)}</SelectContent>
              </Select>
              <Card className="px-3 py-2 text-sm bg-muted/50">
                {leftAlgo?.parameters.map((p) => (
                  <div key={p.name}>{p.name} = {leftParams[p.name] || p.defaultValue}</div>
                ))}
              </Card>
            </div>
            <div className="flex items-center gap-3">
              <Select value={rightAlgorithm} onValueChange={setRightAlgorithm}>
                <SelectTrigger className="h-10 w-48 text-base">
                  <SelectValue />
                  <ChevronDown className="h-4 w-4 ml-2" />
                </SelectTrigger>
                <SelectContent>{algorithms.map((a) => <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>)}</SelectContent>
              </Select>
              <Card className="px-3 py-2 text-sm bg-muted/50">
                {rightAlgo?.parameters.map((p) => (
                  <div key={p.name}>{p.name} = {rightParams[p.name] || p.defaultValue}</div>
                ))}
              </Card>
            </div>
          </div>

          {/* Charts */}
          <div className="grid lg:grid-cols-2 gap-3 overflow-hidden">
            <AccuracyChart algorithmName={leftAlgo?.name || "Algorithm 1"} />
            <AccuracyChart algorithmName={rightAlgo?.name || "Algorithm 2"} />
          </div>

          {/* Demo button */}
          <Button onClick={() => navigateTo("demo")} size="lg" className="shrink-0">
            Demo
          </Button>
        </div>
      )}
    </div>
  )
}
