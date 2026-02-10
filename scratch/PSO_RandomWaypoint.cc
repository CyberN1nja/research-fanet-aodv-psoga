/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"

#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>
#include <set>
#include <numeric>
#include <limits>

using namespace ns3;

uint32_t currentRound = 0;

NS_LOG_COMPONENT_DEFINE("FANET_PSO_ONLY");

struct FlowStats {
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    double totalDelay = 0.0;
    double totalRssi = 0.0;
    uint32_t rssiCount = 0;
    double totalPdr = 0.0;
    uint32_t pdrCount = 0;
};

/* ====================== PSO-ONLY OPTIMIZER ====================== */
class PSOOptimizer {
public:
    struct RouteSolution {
        std::vector<uint32_t> path;
        double fitness = 0.0;
        double avgRssi = 0.0;
        double avgLatency = 0.0;
        double avgPdr = 0.0;
        double totalDistance = 0.0;
    };

    struct SwapOp { size_t i, j; };          // velocity = daftar swap (perm-discrete)
    using Velocity = std::vector<SwapOp>;

    struct Particle {
        RouteSolution current;
        Velocity velocity;
        RouteSolution personalBest;
    };

    PSOOptimizer(uint32_t numParticles, uint32_t maxIter, double c1, double c2, double w,
                 double convThreshold = 0.001, uint32_t noImproveIter = 10)
        : numParticles(numParticles), maxIterations(maxIter),
          c1(c1), c2(c2), inertiaWeight(w),
          convergenceThreshold(convThreshold), noImprovementIterations(noImproveIter) {}

    void Initialize(uint32_t source, uint32_t destination, uint32_t numNodes,
                    const std::map<std::pair<uint32_t, uint32_t>, double>& rssiMap,
                    const std::map<std::pair<uint32_t, uint32_t>, double>& latencyMap,
                    const std::map<std::pair<uint32_t, uint32_t>, double>& distanceMap,
                    const std::map<std::pair<uint32_t, uint32_t>, double>& pdrMap) {
        this->source = source;
        this->destination = destination;
        this->numNodes = numNodes;
        this->rssiMap = rssiMap;
        this->latencyMap = latencyMap;
        this->distanceMap = distanceMap;
        this->pdrMap = pdrMap;
        this->currentIteration = 0;
        this->noImprovementCount = 0;
        this->previousBestFitness = -1e18;

        std::random_device rd;
        std::mt19937 gen(rd());

        particles.clear();
        particles.resize(numParticles);
        for (auto& p : particles) {
            p.current.path = GenerateRandomRoute(gen);
            p.current = EvaluateFitness(p.current.path);
            p.personalBest = p.current;

            // seed velocity: 1..3 swap pada bagian tengah
            p.velocity.clear();
            if (p.current.path.size() > 3) {
                std::uniform_int_distribution<> posDist(1, (int)p.current.path.size() - 2);
                size_t seedSwaps = 1 + (gen() % 3);
                for (size_t s = 0; s < seedSwaps; ++s) {
                    size_t i = posDist(gen), j = posDist(gen);
                    if (i == j) continue;
                    if (i > j) std::swap(i, j);
                    p.velocity.push_back({i, j});
                }
            }
        }

        globalBest = particles[0].personalBest;
        for (size_t i = 1; i < particles.size(); ++i) {
            if (particles[i].personalBest.fitness > globalBest.fitness)
                globalBest = particles[i].personalBest;
        }
    }

    uint32_t Optimize() {
        for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration) {
            // Update velocity & position semua partikel
            for (auto& p : particles) UpdateVelocity(p);
            for (auto& p : particles) UpdatePosition(p);

            // Evaluasi & update pBest/gBest
            for (auto& p : particles) {
                p.current = EvaluateFitness(p.current.path);
                if (p.current.fitness > p.personalBest.fitness) p.personalBest = p.current;
                if (p.current.fitness > globalBest.fitness)      globalBest     = p.current;
            }

            if (CheckConvergence()) {
                NS_LOG_INFO("PSO converged at iteration " << currentIteration);
                break;
            }
        }
        return currentIteration + 1;
    }

    RouteSolution GetBestSolution() const { return globalBest; }
    uint32_t GetCurrentIteration() const { return currentIteration; }

private:
    /* ---------- Konvergensi ---------- */
    bool CheckConvergence() {
        double improvement = globalBest.fitness - previousBestFitness;
        if (improvement < convergenceThreshold) ++noImprovementCount;
        else noImprovementCount = 0;
        previousBestFitness = globalBest.fitness;
        return (noImprovementCount >= noImprovementIterations);
    }

    /* ---------- Fitness & Normalisasi ---------- */
    double CalculateCompositeFitness(double avgRssi, double avgLatency, double avgPdr) {
        double normRssi = NormalizeRSSI(avgRssi);
        double normLatency = NormalizeLatency(avgLatency);
        double normPdr = std::clamp(avgPdr / 100.0, 0.0, 1.0);
        return (0.7 * normRssi) + (0.2 * normLatency) + (0.1 * normPdr);
    }

    double NormalizeRSSI(double rssi) {
        return std::max(0.0, std::min(1.0, (rssi + 110.0) / 40.0));
    }

    double NormalizeLatency(double latency) {
        // small latency is better; 0.1s as reference scale
        return std::max(0.0, 1.0 - (latency / 0.1));
    }

    /* ---------- Generator rute acak ---------- */
    std::vector<uint32_t> GenerateRandomRoute(std::mt19937& gen) {
        std::vector<uint32_t> path = {source};
        std::uniform_int_distribution<> dist(0, numNodes - 1);
        int attempts = 0;
        const int maxAttempts = (int)numNodes * 3;

        while (path.back() != destination && attempts < maxAttempts) {
            uint32_t nextNode = dist(gen);
            if (nextNode != path.back() &&
                std::find(path.begin(), path.end(), nextNode) == path.end()) {
                path.push_back(nextNode);
            }
            attempts++;
        }
        if (path.back() != destination) path.push_back(destination);

        std::set<uint32_t> visited;
        std::vector<uint32_t> cleanedPath;
        for (uint32_t node : path) {
            if (visited.insert(node).second) cleanedPath.push_back(node);
        }
        if (!cleanedPath.empty()) {
            cleanedPath.front() = source;
            cleanedPath.back()  = destination;
        }
        return cleanedPath;
    }

    /* ---------- Evaluasi rute ---------- */
    RouteSolution EvaluateFitness(const std::vector<uint32_t>& path) {
        std::set<uint32_t> uniqueNodes(path.begin(), path.end());
        if (uniqueNodes.size() != path.size()) {
            return {path, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
        if (path.size() < 2 || path.front() != source || path.back() != destination) {
            return {path, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        RouteSolution r;
        r.path = path;
        double totalRssi = 0.0, totalLatency = 0.0, totalPdr = 0.0, totalDistance = 0.0;
        int hopCount = 0;

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            uint32_t u = path[i], v = path[i + 1];
            auto key = std::make_pair(u, v);

            if (rssiMap.find(key) != rssiMap.end()) {
                totalRssi    += rssiMap.at(key);
                totalLatency += latencyMap.count(key) ? latencyMap.at(key) : 0.1;
                totalPdr     += pdrMap.count(key) ? pdrMap.at(key) : 50.0;
            } else {
                totalRssi    += -90.0;
                totalLatency += 0.1;
                totalPdr     += 30.0;
            }

            if (distanceMap.find(key) != distanceMap.end())
                totalDistance += distanceMap.at(key);
            ++hopCount;
        }

        if (hopCount == 0) return r;
        r.avgRssi       = totalRssi / hopCount;
        r.avgLatency    = totalLatency / hopCount;
        r.avgPdr        = totalPdr / hopCount;
        r.totalDistance = totalDistance;
        r.fitness       = CalculateCompositeFitness(r.avgRssi, r.avgLatency, r.avgPdr);
        return r;
    }

    /* ---------- Discrete PSO ---------- */

    // beda dua permutasi (bagian tengah) → daftar swap
    Velocity DiffPermutation(const std::vector<uint32_t>& from, const std::vector<uint32_t>& to) {
        Velocity ops;
        if (from.size() != to.size() || from.size() < 3) return ops;

        std::vector<uint32_t> a = from;
        for (size_t pos = 1; pos + 1 < a.size(); ++pos) {
            if (a[pos] == to[pos]) continue;
            size_t q = pos + 1;
            while (q + 1 < a.size() && a[q] != to[pos]) ++q;
            if (q >= a.size()) continue;
            ops.push_back({pos, q});
            std::swap(a[pos], a[q]);
        }
        return ops;
    }

    // gabungkan v_old + tarikan pBest + tarikan gBest
    Velocity CombineVelocities(const Velocity& vOld, const Velocity& vP, const Velocity& vG,
                               double keepOldRate, double takePFrac, double takeGFrac,
                               size_t vMax) {
        Velocity v;
        size_t kOld = static_cast<size_t>(std::round(keepOldRate * vOld.size()));
        kOld = std::min(kOld, vOld.size());
        v.insert(v.end(), vOld.begin(), vOld.begin() + kOld);

        size_t kP = static_cast<size_t>(std::round(takePFrac * vP.size()));
        kP = std::min(kP, vP.size());
        v.insert(v.end(), vP.begin(), vP.begin() + kP);

        size_t kG = static_cast<size_t>(std::round(takeGFrac * vG.size()));
        kG = std::min(kG, vG.size());
        v.insert(v.end(), vG.begin(), vG.begin() + kG);

        if (v.size() > vMax) v.resize(vMax);
        return v;
    }

    // terapkan K swap pertama
    void ApplyVelocity(std::vector<uint32_t>& path, const Velocity& vel, size_t K) {
        if (path.size() < 3) return;
        K = std::min(K, vel.size());
        for (size_t t = 0; t < K; ++t) {
            size_t i = vel[t].i, j = vel[t].j;
            if (i == 0 || j == 0) continue;
            if (i + 1 == path.size() || j + 1 == path.size()) continue;
            if (i >= path.size() || j >= path.size() || i == j) continue;
            std::swap(path[i], path[j]);
        }
    }

    // repair: jaga source/dest & hapus duplikat tengah
    void RepairPath(std::vector<uint32_t>& path, uint32_t source, uint32_t destination) {
        if (path.empty()) return;
        path.front() = source;
        path.back()  = destination;
        std::set<uint32_t> seen;
        std::vector<uint32_t> cleaned;
        cleaned.reserve(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            uint32_t node = path[i];
            if (i == 0 || i + 1 == path.size()) {
                cleaned.push_back(node);
            } else if (node != source && node != destination && seen.insert(node).second) {
                cleaned.push_back(node);
            }
        }
        if (cleaned.size() < 2) {
            cleaned.clear();
            cleaned.push_back(source);
            cleaned.push_back(destination);
        }
        path.swap(cleaned);
    }

    void UpdateVelocity(Particle& p) {
        const size_t vMax = std::max<size_t>(5, numNodes);

        auto clamp01 = [](double x){ return std::max(0.0, std::min(1.0, x)); };
        double takePFrac   = clamp01(c1 / (c1 + c2 + 1e-9));
        double takeGFrac   = clamp01(c2 / (c1 + c2 + 1e-9));
        double keepOldRate = clamp01(inertiaWeight);

        Velocity vP = DiffPermutation(p.current.path, p.personalBest.path);
        Velocity vG = DiffPermutation(p.current.path, globalBest.path);

        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_real_distribution<> ur(0.7, 1.0);   // sedikit stochastik
        takePFrac   *= ur(gen);
        takeGFrac   *= ur(gen);
        keepOldRate *= ur(gen);

        p.velocity = CombineVelocities(p.velocity, vP, vG, keepOldRate, takePFrac, takeGFrac, vMax);
    }

    void UpdatePosition(Particle& p) {
        if (p.current.path.size() < 3) return;
        std::random_device rd; std::mt19937 gen(rd());
        std::uniform_real_distribution<> stepUr(0.5, 1.0);
        size_t K = static_cast<size_t>(std::round(stepUr(gen) * p.velocity.size()));
        std::vector<uint32_t> newPath = p.current.path;
        ApplyVelocity(newPath, p.velocity, K);
        RepairPath(newPath, source, destination);
        p.current.path = newPath;
    }

private:
    uint32_t numParticles;
    uint32_t maxIterations;
    uint32_t currentIteration = 0;
    double c1, c2, inertiaWeight;
    double convergenceThreshold;
    uint32_t noImprovementIterations;
    uint32_t noImprovementCount = 0;
    double previousBestFitness = 0.0;

    std::vector<Particle> particles;
    RouteSolution globalBest;

    uint32_t source = 0;
    uint32_t destination = 0;
    uint32_t numNodes = 0;

    std::map<std::pair<uint32_t, uint32_t>, double> rssiMap;
    std::map<std::pair<uint32_t, uint32_t>, double> latencyMap;
    std::map<std::pair<uint32_t, uint32_t>, double> distanceMap;
    std::map<std::pair<uint32_t, uint32_t>, double> pdrMap;
};
/* ==================== END PSO-ONLY OPTIMIZER ==================== */

/* ====================== BAGIAN SIMULASI NS-3 ====================== */
Ptr<FlowMonitor> flowMonitor;
std::map<uint32_t, std::ostringstream> nodeDataBuffers;
std::map<std::pair<uint32_t, uint32_t>, FlowStats> flowStatsMap;
std::map<uint64_t, std::pair<double, uint32_t>> packetTxInfo;
std::map<std::pair<uint32_t, uint32_t>, double> rssiMeasurements;
std::map<std::pair<uint32_t, uint32_t>, double> latencyMeasurements;
std::map<std::pair<uint32_t, uint32_t>, double> distanceMeasurements;
std::map<std::pair<uint32_t, uint32_t>, double> pdrMeasurements;
std::map<uint64_t, double> packetProcessingStartTimes;
double maxCommunicationRange = 1000.0;

void TransmitPacketCallback(std::string context, Ptr<const Packet> packet, double txPowerDbm) {
    std::string nodeStr = context.substr(10);
    size_t devicePos = nodeStr.find("/Device");
    uint32_t nodeId = std::stoi(nodeStr.substr(0, devicePos));

    double currentTime = Simulator::Now().GetSeconds();
    packetTxInfo[packet->GetUid()] = std::make_pair(currentTime, nodeId);

    for (auto& entry : flowStatsMap) {
        if (entry.first.first == nodeId) entry.second.txPackets++;
    }
}

void ReceivePacketCallback(std::string context, Ptr<const Packet> packet,
                           uint16_t channelFreqMhz, WifiTxVector txVector,
                           MpduInfo mpduInfo, SignalNoiseDbm signalNoise,
                           uint16_t staId) {

    Ptr<Packet> copy = packet->Copy();
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    if (copy->RemoveHeader(ipHeader) && ipHeader.GetProtocol() == 17) {
        copy->RemoveHeader(udpHeader); // isAppPacket tidak dipakai, tapi parsing aman
    }

    std::string nodeStr = context.substr(10);
    size_t devicePos = nodeStr.find("/Device");
    uint32_t receiverNodeId = std::stoi(nodeStr.substr(0, devicePos));

    double rxTime = Simulator::Now().GetSeconds();

    auto txInfoIt = packetTxInfo.find(packet->GetUid());
    if (txInfoIt == packetTxInfo.end()) return;

    double txTime = txInfoIt->second.first;
    uint32_t senderNodeId = txInfoIt->second.second;
    packetTxInfo.erase(txInfoIt);

    if (senderNodeId == receiverNodeId) return;

    // Jarak
    double distance = -1.0;
    Ptr<Node> senderNode = NodeList::GetNode(senderNodeId);
    Ptr<Node> receiverNode = NodeList::GetNode(receiverNodeId);
    if (senderNode && receiverNode) {
        Ptr<MobilityModel> sm = senderNode->GetObject<MobilityModel>();
        Ptr<MobilityModel> rm = receiverNode->GetObject<MobilityModel>();
        if (sm && rm) {
            distance = CalculateDistance(sm->GetPosition(), rm->GetPosition());
            distanceMeasurements[std::make_pair(senderNodeId, receiverNodeId)] = distance;
        }
    }

    // Delay komponen
    double propagationDelay = (distance > 0) ? (distance / 3e8) : 0.0;
    double transmissionDelay = packet->GetSize() * 8.0 / txVector.GetMode().GetDataRate(txVector);

    double queueProcessingDelay = 0.0;
    auto procIt = packetProcessingStartTimes.find(packet->GetUid());
    if (procIt != packetProcessingStartTimes.end()) {
        double processingStartTime = procIt->second;
        queueProcessingDelay = rxTime - processingStartTime - propagationDelay - transmissionDelay;
        queueProcessingDelay = std::max(0.0, queueProcessingDelay);
        packetProcessingStartTimes.erase(procIt);
    }

    double perHopDelay = propagationDelay + transmissionDelay + queueProcessingDelay;

    auto flowKey = std::make_pair(senderNodeId, receiverNodeId);
    FlowStats& flowStats = flowStatsMap[flowKey];

    flowStats.rxPackets++;
    flowStats.totalDelay += perHopDelay;
    flowStats.totalRssi += signalNoise.signal;
    flowStats.rssiCount++;

    double pdr = (flowStats.txPackets > 0) ?
        (static_cast<double>(flowStats.rxPackets) / flowStats.txPackets) * 100.0 : 0.0;
    pdr = std::min(100.0, std::max(0.0, pdr));

    if (flowStats.txPackets > 0) {
        double currentPdr = (static_cast<double>(flowStats.rxPackets) / flowStats.txPackets) * 100.0;
        currentPdr = std::min(100.0, std::max(0.0, currentPdr));
        flowStats.totalPdr += currentPdr;
        flowStats.pdrCount++;
        pdrMeasurements[flowKey] = flowStats.totalPdr / flowStats.pdrCount;
    }

    std::ostringstream perHopDelayStr, propDelayStr, transDelayStr, queueDelayStr;
    perHopDelayStr << std::fixed << std::setprecision(9) << perHopDelay;
    propDelayStr  << std::fixed << std::setprecision(9) << propagationDelay;
    transDelayStr << std::fixed << std::setprecision(9) << transmissionDelay;
    queueDelayStr << std::fixed << std::setprecision(9) << queueProcessingDelay;

    nodeDataBuffers[receiverNodeId] << "  <Record>\n"
        << "    <Sender>" << senderNodeId << "</Sender>\n"
        << "    <Receiver>" << receiverNodeId << "</Receiver>\n"
        << "    <Time_Seconds>" << std::fixed << std::setprecision(3) << rxTime << "</Time_Seconds>\n"
        << "    <RSSI_dBm>" << signalNoise.signal << "</RSSI_dBm>\n"
        << "    <PerHop_Delay>" << perHopDelayStr.str() << "</PerHop_Delay>\n"
        << "    <Propagation_Delay>" << propDelayStr.str() << "</Propagation_Delay>\n"
        << "    <Transmission_Delay>" << transDelayStr.str() << "</Transmission_Delay>\n"
        << "    <QueueProcessing_Delay>" << queueDelayStr.str() << "</QueueProcessing_Delay>\n"
        << "    <Flow_PDR_Percent>" << std::fixed << std::setprecision(2) << pdr << "</Flow_PDR_Percent>\n"
        << "    <Distance_Meters>" << std::fixed << std::setprecision(2) << distance << "</Distance_Meters>\n"
        << "    <Sent_Packets>" << flowStats.txPackets << "</Sent_Packets>\n"
        << "    <Received_Packets>" << flowStats.rxPackets << "</Received_Packets>\n"
        << "  </Record>\n";

    rssiMeasurements[flowKey] = flowStats.rssiCount > 0 ?
        flowStats.totalRssi / flowStats.rssiCount : signalNoise.signal;
    latencyMeasurements[flowKey] = perHopDelay;
}

void PrintFlowStatistics(uint32_t numNodes) {
    for (uint32_t i = 0; i < numNodes; ++i) {
        uint32_t totalTx = 0, totalRx = 0;
        double totalDelay = 0.0;
        uint32_t flowCount = 0;
        for (auto& entry : flowStatsMap) {
            uint32_t sender = entry.first.first;
            uint32_t receiver = entry.first.second;
            FlowStats& stats = entry.second;
            if (sender == i) totalTx += stats.txPackets;
            if (receiver == i) {
                totalRx += stats.rxPackets;
                if (stats.rxPackets > 0) {
                    totalDelay += stats.totalDelay;
                    flowCount++;
                }
            }
        }
        (void)totalTx; (void)totalRx; (void)totalDelay; (void)flowCount;
    }
}

void SaveNodeData(uint32_t numNodes) {
    for (uint32_t i = 0; i < numNodes; ++i) {
        uint32_t totalTx = 0, totalRx = 0;
        double totalDelay = 0.0;
        uint32_t validFlows = 0;

        for (const auto& entry : flowStatsMap) {
            uint32_t sender = entry.first.first;
            uint32_t receiver = entry.first.second;
            const FlowStats& stats = entry.second;

            if (sender == i) totalTx += stats.txPackets;
            if (receiver == i) {
                totalRx += stats.rxPackets;
                if (stats.rxPackets > 0) {
                    totalDelay += stats.totalDelay;
                    validFlows++;
                }
            }
        }

        double avgDelay = (validFlows > 0) ? totalDelay / validFlows : 0.0;
        double pdr = (totalTx > 0) ? (static_cast<double>(totalRx) / totalTx) * 100.0 : 0.0;
        pdr = std::min(100.0, std::max(0.0, pdr));

        if (nodeDataBuffers[i].str().find("<?xml") == std::string::npos) {
            std::ostringstream tempBuffer;
            tempBuffer << "<?xml version=\"1.0\"?>\n<Node id=\"" << i << "\">\n";
            tempBuffer << nodeDataBuffers[i].str();
            nodeDataBuffers[i].str("");
            nodeDataBuffers[i] << tempBuffer.str();
        }

        nodeDataBuffers[i] << "  <Summary>\n"
            << "    <TX_Packets>" << totalTx << "</TX_Packets>\n"
            << "    <RX_Packets>" << totalRx << "</RX_Packets>\n"
            << "    <Delay_Seconds>" << avgDelay << "</Delay_Seconds>\n"
            << "    <PDR_Percent>" << pdr << "</PDR_Percent>\n"
            << "  </Summary>\n"
            << "</Node>\n";

        std::string filePath = "/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_PSO/simulasi_dengan_30_node/20/node_" + std::to_string(i) + "_data.xml";
        std::ofstream outFile(filePath);
        if (outFile.is_open()) {
            outFile << nodeDataBuffers[i].str();
            outFile.close();
        }
    }
}

void LogNodePositions(Ptr<Node> node, uint32_t nodeId, double interval) {
    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    if (mob) { mob->GetPosition(); }
    Simulator::Schedule(Seconds(interval), &LogNodePositions, node, nodeId, interval);
}

void EnqueuePacketCallback(std::string context, Ptr<const WifiMpdu> mpdu) {
    if (mpdu) {
        double enqueueTime = Simulator::Now().GetSeconds();
        packetProcessingStartTimes[mpdu->GetPacket()->GetUid()] = enqueueTime;
    }
}

int main(int argc, char *argv[]) {
    LogComponentEnable("FANET_PSO_ONLY", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

    uint32_t numUAVs = 100;
    double simulationTime = 300.0;
    double optimizationInterval = 15.0;

    NodeContainer uavNodes;
    uavNodes.Create(numUAVs);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(maxCommunicationRange));

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::MinstrelWifiManager");

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devices = wifi.Install(phy, mac, uavNodes);

    if (!Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                                 MakeCallback(&TransmitPacketCallback))) {
        NS_FATAL_ERROR("Failed to register TX callback");
    }
    if (!Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/MonitorSnifferRx",
                                 MakeCallback(&ReceivePacketCallback))) {
        NS_FATAL_ERROR("Failed to register RX callback");
    }
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/Txop/Queue/Enqueue",
                    MakeCallback(&EnqueuePacketCallback));

    MobilityHelper mobility;
    // Random Waypoint (3D) - area 5000x5000 dan tinggi 0–2000
    Ptr<RandomBoxPositionAllocator> positionAlloc = CreateObject<RandomBoxPositionAllocator>();
    positionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"));
    positionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"));
    positionAlloc->SetAttribute("Z", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=2000.0]"));


    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                              "Speed", StringValue("ns3::UniformRandomVariable[Min=38.9|Max=44.4]"),
                              "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
                              "PositionAllocator", PointerValue(positionAlloc));

    mobility.Install(uavNodes);

    AodvHelper aodv;
    aodv.Set("EnableHello", BooleanValue(true));
    aodv.Set("HelloInterval", TimeValue(Seconds(1)));
    aodv.Set("AllowedHelloLoss", UintegerValue(2));
    aodv.Set("ActiveRouteTimeout", TimeValue(Seconds(10)));

    InternetStackHelper internetWithAodv;
    internetWithAodv.SetRoutingHelper(aodv);

    NodeContainer nodesWithoutNode0;
    for (uint32_t i = 1; i < uavNodes.GetN(); ++i) nodesWithoutNode0.Add(uavNodes.Get(i));
    internetWithAodv.Install(nodesWithoutNode0);

    InternetStackHelper internetNoRouting;
    internetNoRouting.Install(uavNodes.Get(0));

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    UdpEchoServerHelper echoServer(9);
    ApplicationContainer serverApps = echoServer.Install(uavNodes.Get(0));
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(simulationTime));

    for (uint32_t i = 1; i < numUAVs; ++i) {
        UdpEchoClientHelper echoClient(interfaces.GetAddress(0), 9);
        echoClient.SetAttribute("MaxPackets", UintegerValue(1500));
        echoClient.SetAttribute("Interval", TimeValue(Seconds(200)));
        echoClient.SetAttribute("PacketSize", UintegerValue(1024));
        ApplicationContainer clientApps = echoClient.Install(uavNodes.Get(i));
        clientApps.Start(Seconds(10.0 + i));
        clientApps.Stop(Seconds(simulationTime));
    }

    for (uint32_t i = 0; i < numUAVs; ++i) {
        nodeDataBuffers[i] << "<?xml version=\"1.0\"?>\n"
                           << "<Node id=\"" << i << "\">\n";
    }

    FlowMonitorHelper flowHelper;
    flowMonitor = flowHelper.InstallAll();

    AnimationInterface anim("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_PSO/simulasi_dengan_30_node/20/fanet-animation.xml");
    anim.SetMobilityPollInterval(Seconds(0.1));
    anim.EnablePacketMetadata(true);

    uint32_t uavImageId = anim.AddResource("/home/rangga/ns-3-dev/uav.svg");
    for (uint32_t i = 0; i < numUAVs; ++i) {
        anim.UpdateNodeImage(uavNodes.Get(i)->GetId(), uavImageId);
        anim.UpdateNodeSize(uavNodes.Get(i)->GetId(), 50.0, 50.0);
    }

    // ======== PSO-ONLY Optimizer (parameter sama) ========
    PSOOptimizer optimizer(30, 50, 2.0, 2.0, 0.6, 0.001, 10);

    for (double t = 60.0; t < simulationTime; t += optimizationInterval) {
        Simulator::Schedule(Seconds(t), [&, t, numUAVs]() {
            currentRound++;

            if (rssiMeasurements.size() < 3 || latencyMeasurements.size() < 3 || pdrMeasurements.size() < 3) {
                return;
            }

            // source = numUAVs-1, dest = 0 (sama seperti program semula)
            optimizer.Initialize(numUAVs - 1, 0, numUAVs,
                                 rssiMeasurements, latencyMeasurements,
                                 distanceMeasurements, pdrMeasurements);

            uint32_t iterationsUsed = optimizer.Optimize();
            auto bestRoute = optimizer.GetBestSolution();

            if (bestRoute.path.size() <= 2) return;

            std::set<uint32_t> uniqueNodes(bestRoute.path.begin(), bestRoute.path.end());
            if (uniqueNodes.size() != bestRoute.path.size()) return;

            for (uint32_t i = 0; i < numUAVs; ++i) {
                nodeDataBuffers[i] << "  <BestRouteResult round=\"" << currentRound << "\">\n"
                                   << "    <BestRoute>";
                for (auto node : bestRoute.path) nodeDataBuffers[i] << node << " ";
                nodeDataBuffers[i] << "</BestRoute>\n"
                                   << "    <Fitness>" << bestRoute.fitness << "</Fitness>\n"
                                   << "    <AvgRSSI>" << bestRoute.avgRssi << "</AvgRSSI>\n"
                                   << "    <AvgLatency>" << bestRoute.avgLatency << "</AvgLatency>\n"
                                   << "    <AvgPDR>" << bestRoute.avgPdr << "</AvgPDR>\n"
                                   << "    <TotalDistance>" << bestRoute.totalDistance << "</TotalDistance>\n"
                                   << "    <Iterations>" << iterationsUsed << "</Iterations>\n"
                                   << "  </BestRouteResult>\n";
            }

            Simulator::Schedule(Seconds(simulationTime - 1), [&]() {
                for (uint32_t i = 0; i < numUAVs; ++i) {
                    nodeDataBuffers[i] << "  <NetworkSummary>\n"
                                       << "    <TotalRounds>" << currentRound << "</TotalRounds>\n"
                                       << "    <SimulationTime>" << simulationTime << "</SimulationTime>\n"
                                       << "  </NetworkSummary>\n";
                }
            });
        });
    }

    for (uint32_t i = 0; i < numUAVs; ++i) {
        Simulator::Schedule(Seconds(0.1), &LogNodePositions, uavNodes.Get(i), i, 1.0);
    }

    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();

    flowMonitor->SerializeToXmlFile("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_PSO/simulasi_dengan_30_node/20/flow-monitor.xml", true, true);

    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats();
    uint32_t totalAppPackets = 0;
    for (auto &flow : stats) {
        if (classifier->FindFlow(flow.first).destinationPort == 9) {
            totalAppPackets += flow.second.txPackets;
        }
    }
    NS_LOG_INFO("Total application packets (port 9): " << totalAppPackets);

    SaveNodeData(numUAVs);

    std::ofstream summaryFile("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_PSO/simulasi_dengan_30_node/20/network_summary.xml");
    summaryFile << "<?xml version=\"1.0\"?>\n<NetworkSummary>\n"
                << "  <TotalRounds>" << currentRound << "</TotalRounds>\n"
                << "  <TotalApplicationPackets>" << totalAppPackets << "</TotalApplicationPackets>\n"
                << "</NetworkSummary>\n";
    summaryFile.close();

    Simulator::Destroy();
    return 0;
}
