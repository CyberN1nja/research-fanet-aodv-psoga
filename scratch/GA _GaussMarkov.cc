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

NS_LOG_COMPONENT_DEFINE("FANET_GA_ONLY");

struct FlowStats {
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    double totalDelay = 0.0;
    double totalRssi = 0.0;
    uint32_t rssiCount = 0;
    double totalPdr = 0.0;  // PDR kumulatif
    uint32_t pdrCount = 0;  // jumlah akumulasi PDR
};

/* ====================== GA-ONLY OPTIMIZER ====================== */
class GAOptimizer {
public:
    struct RouteSolution {
        std::vector<uint32_t> path;
        double fitness = 0.0;
        double avgRssi = 0.0;
        double avgLatency = 0.0;
        double avgPdr = 0.0;
        double totalDistance = 0.0;
    };

    struct Individual {
        RouteSolution current;
    };

    GAOptimizer(uint32_t numParticles, uint32_t maxIter,
                double mutationProb, double elitismRate, double crossoverProb,
                double convThreshold = 0.001, uint32_t noImproveIter = 10)
        : numParticles(numParticles), maxIterations(maxIter),
          mutationProb(mutationProb), elitismRate(elitismRate), crossoverProb(crossoverProb),
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

        std::random_device rd; std::mt19937 gen(rd());

        pop.clear();
        pop.resize(numParticles);
        for (auto& ind : pop) {
            ind.current.path = GenerateRandomRoute(gen);
            ind.current = EvaluateFitness(ind.current.path);
        }

        globalBest = pop[0].current;
        for (size_t i = 1; i < pop.size(); ++i)
            if (pop[i].current.fitness > globalBest.fitness) globalBest = pop[i].current;
    }

    uint32_t Optimize() {
        std::random_device rd; std::mt19937 gen(rd());
        for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration) {
            // 1) Sort by fitness (desc)
            std::sort(pop.begin(), pop.end(),
                [](const Individual& a, const Individual& b){ return a.current.fitness > b.current.fitness; });

            // 2) Elitism
            size_t eliteCount = EliteCount();
            std::vector<Individual> nextGen(pop.begin(), pop.begin() + eliteCount);

            // 3) Fill the rest via selection + crossover + mutation
            std::uniform_real_distribution<> ur(0.0, 1.0);
            while (nextGen.size() < pop.size()) {
                size_t pA = RoulettePick(pop, 0, pop.size(), gen);
                size_t pB;
                do { pB = RoulettePick(pop, 0, pop.size(), gen); } while (pB == pA);

                Individual child1 = pop[pA];
                Individual child2 = pop[pB];

                if (ur(gen) < crossoverProb) {
                    ArithmeticCrossover(child1, child2, gen);
                }

                // Mutasi hanya untuk non-elit yang baru dibuat
                Mutate(child1, gen);
                Mutate(child2, gen);

                // Evaluate anak
                child1.current = EvaluateFitness(child1.current.path);
                child2.current = EvaluateFitness(child2.current.path);

                nextGen.push_back(child1);
                if (nextGen.size() < pop.size()) nextGen.push_back(child2);
            }

            pop.swap(nextGen);

            // 4) Update global best
            for (auto& ind : pop) {
                if (ind.current.fitness > globalBest.fitness) globalBest = ind.current;
            }

            // 5) Check convergence
            if (CheckConvergence()) {
                NS_LOG_INFO("GA converged at iteration " << currentIteration);
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
        // Semakin kecil latency semakin baik; 0.1 s sebagai skala
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

        RouteSolution result;
        result.path = path;

        double totalRssi = 0.0;
        double totalLatency = 0.0;
        double totalPdr = 0.0;
        double totalDistance = 0.0;
        int hopCount = 0;

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            uint32_t from = path[i];
            uint32_t to   = path[i + 1];
            auto key = std::make_pair(from, to);

            if (rssiMap.find(key) != rssiMap.end()) {
                totalRssi   += rssiMap.at(key);
                totalLatency+= latencyMap.count(key) ? latencyMap.at(key) : 0.1;
                totalPdr    += pdrMap.count(key) ? pdrMap.at(key) : 50.0;
            } else {
                // estimasi konservatif
                totalRssi   += -90.0;
                totalLatency+= 0.1;
                totalPdr    += 30.0;
            }

            if (distanceMap.find(key) != distanceMap.end()) {
                totalDistance += distanceMap.at(key);
            }
            ++hopCount;
        }

        if (hopCount == 0) return result;

        result.avgRssi    = totalRssi / hopCount;
        result.avgLatency = totalLatency / hopCount;
        result.avgPdr     = totalPdr / hopCount;
        result.totalDistance = totalDistance;
        result.fitness = CalculateCompositeFitness(result.avgRssi, result.avgLatency, result.avgPdr);
        return result;
    }

    /* ---------- GA helpers ---------- */
    size_t EliteCount() const {
        size_t e = static_cast<size_t>(std::round(elitismRate * numParticles));
        return std::min(e, static_cast<size_t>(numParticles));
    }

    // Roulette-wheel selection dengan shifting untuk fitness negatif
    size_t RoulettePick(const std::vector<Individual>& population,
                        size_t startIdx, size_t endIdx, std::mt19937& gen) {
        double minFit = std::numeric_limits<double>::infinity();
        for (size_t i = startIdx; i < endIdx; ++i)
            minFit = std::min(minFit, population[i].current.fitness);

        const double eps = 1e-12;
        double total = 0.0;
        std::vector<double> w(endIdx - startIdx);
        for (size_t i = startIdx; i < endIdx; ++i) {
            double shifted = population[i].current.fitness - (minFit < 0 ? minFit : 0.0);
            double wi = std::max(shifted, 0.0);
            w[i - startIdx] = wi;
            total += wi;
        }

        std::uniform_real_distribution<> ur(0.0, 1.0);
        if (total <= eps) {
            std::uniform_int_distribution<> uni((int)startIdx, (int)endIdx - 1);
            return (size_t)uni(gen);
        }

        double r = ur(gen) * total;
        double acc = 0.0;
        for (size_t i = startIdx; i < endIdx; ++i) {
            acc += w[i - startIdx];
            if (r <= acc) return i;
        }
        return endIdx - 1;
    }

    // Repair path: source/dest fix, hilangkan duplikat tengah
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

    // Arithmetic crossover berbasis priority-key untuk permutasi (bagian tengah)
    void ArithmeticCrossover(Individual& p1, Individual& p2, std::mt19937& gen) {
        if (p1.current.path.size() < 3 || p2.current.path.size() < 3) return;

        std::vector<uint32_t> mid1(p1.current.path.begin() + 1, p1.current.path.end() - 1);
        std::vector<uint32_t> mid2(p2.current.path.begin() + 1, p2.current.path.end() - 1);
        if (mid1.empty() || mid2.empty()) return;

        std::set<uint32_t> Uset(mid1.begin(), mid1.end());
        Uset.insert(mid2.begin(), mid2.end());
        std::vector<uint32_t> U(Uset.begin(), Uset.end());

        const double BIG = 1e6;
        std::map<uint32_t, double> k1, k2;

        for (uint32_t v : U) {
            auto it1 = std::find(mid1.begin(), mid1.end(), v);
            auto it2 = std::find(mid2.begin(), mid2.end(), v);
            double pos1 = (it1 == mid1.end()) ? BIG : (double)std::distance(mid1.begin(), it1);
            double pos2 = (it2 == mid2.end()) ? BIG : (double)std::distance(mid2.begin(), it2);

            double n1 = (pos1 >= BIG) ? 1.0 : (mid1.size() > 1 ? pos1 / (mid1.size() - 1.0) : 0.0);
            double n2 = (pos2 >= BIG) ? 1.0 : (mid2.size() > 1 ? pos2 / (mid2.size() - 1.0) : 0.0);
            k1[v] = n1; k2[v] = n2;
        }

        std::uniform_real_distribution<> ur(0.25, 0.75);
        double alpha = ur(gen);

        size_t K1 = (size_t)std::round(alpha * mid1.size() + (1.0 - alpha) * mid2.size());
        size_t K2 = (size_t)std::round((1.0 - alpha) * mid1.size() + alpha * mid2.size());
        K1 = std::max<size_t>(1, std::min(K1, U.size()));
        K2 = std::max<size_t>(1, std::min(K2, U.size()));

        std::vector<std::pair<uint32_t,double>> child1Pairs, child2Pairs;
        child1Pairs.reserve(U.size()); child2Pairs.reserve(U.size());
        for (uint32_t v : U) {
            double key1 = alpha * k1[v] + (1.0 - alpha) * k2[v];
            double key2 = (1.0 - alpha) * k1[v] + alpha * k2[v];
            child1Pairs.emplace_back(v, key1);
            child2Pairs.emplace_back(v, key2);
        }

        std::sort(child1Pairs.begin(), child1Pairs.end(),
                  [](auto& a, auto& b){ return a.second < b.second; });
        std::sort(child2Pairs.begin(), child2Pairs.end(),
                  [](auto& a, auto& b){ return a.second < b.second; });

        std::vector<uint32_t> child1Mid, child2Mid;
        for (size_t i = 0; i < K1 && i < child1Pairs.size(); ++i) child1Mid.push_back(child1Pairs[i].first);
        for (size_t i = 0; i < K2 && i < child2Pairs.size(); ++i) child2Mid.push_back(child2Pairs[i].first);

        std::vector<uint32_t> c1; c1.reserve(child1Mid.size() + 2);
        c1.push_back(source); c1.insert(c1.end(), child1Mid.begin(), child1Mid.end()); c1.push_back(destination);

        std::vector<uint32_t> c2; c2.reserve(child2Mid.size() + 2);
        c2.push_back(source); c2.insert(c2.end(), child2Mid.begin(), child2Mid.end()); c2.push_back(destination);

        RepairPath(c1, source, destination);
        RepairPath(c2, source, destination);

        if (c1.front() == source && c1.back() == destination && c1.size() >= 2)
            p1.current.path = c1;
        if (c2.front() == source && c2.back() == destination && c2.size() >= 2)
            p2.current.path = c2;
    }

    void Mutate(Individual& ind, std::mt19937& gen) {
        auto& path = ind.current.path;
        if (path.size() < 3) return;

        std::uniform_real_distribution<> ur(0.0, 1.0);
        std::uniform_int_distribution<> posDist(1, (int)path.size() - 2);

        for (size_t i = 1; i + 1 < path.size(); ++i) {
            if (ur(gen) < mutationProb) {
                size_t j = posDist(gen);
                if (j == i) continue;
                std::swap(path[i], path[j]);
            }
        }
        RepairPath(path, source, destination);
    }

private:
    uint32_t numParticles;
    uint32_t maxIterations;
    uint32_t currentIteration = 0;
    double mutationProb;
    double elitismRate;
    double crossoverProb;
    double convergenceThreshold;
    uint32_t noImprovementIterations;
    uint32_t noImprovementCount = 0;
    double previousBestFitness = 0.0;

    std::vector<Individual> pop;
    RouteSolution globalBest;

    uint32_t source = 0;
    uint32_t destination = 0;
    uint32_t numNodes = 0;

    std::map<std::pair<uint32_t, uint32_t>, double> rssiMap;
    std::map<std::pair<uint32_t, uint32_t>, double> latencyMap;
    std::map<std::pair<uint32_t, uint32_t>, double> distanceMap;
    std::map<std::pair<uint32_t, uint32_t>, double> pdrMap;
};
/* ==================== END GA-ONLY OPTIMIZER ==================== */

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
    bool isAppPacket = false;

    if (copy->RemoveHeader(ipHeader) && ipHeader.GetProtocol() == 17) { // UDP
        if (copy->RemoveHeader(udpHeader)) {
            isAppPacket = (udpHeader.GetDestinationPort() == 9);
        }
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
        Ptr<MobilityModel> senderMobility = senderNode->GetObject<MobilityModel>();
        Ptr<MobilityModel> receiverMobility = receiverNode->GetObject<MobilityModel>();
        if (senderMobility && receiverMobility) {
            distance = CalculateDistance(senderMobility->GetPosition(),
                                         receiverMobility->GetPosition());
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

        std::string filePath = "/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_GA/simulasi_dengan_30_node/20/node_" + std::to_string(i) + "_data.xml";
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
    LogComponentEnable("FANET_GA_ONLY", LOG_LEVEL_INFO);
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

    // Initial position allocator (random inside the same bounds)
    mobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                  "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"),
                                  "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"),
                                  "Z", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));

    mobility.SetMobilityModel("ns3::GaussMarkovMobilityModel",
                                // Bounds: X[0..5000], Y[0..5000], Z[0..100]
                              "Bounds", BoxValue(Box(0.0, 5000.0, 0.0, 5000.0, 0.0, 100.0)),
                                // Update step
                              "TimeStep", TimeValue(Seconds(0.5)),
                                // Memory factor (0..1). Closer to 1 => smoother trajectories
                              "Alpha", DoubleValue(0.85),
                                // Mean speed around mid of 38.9..44.4 m/s (≈ 150 km/h)
                              "MeanVelocity", StringValue("ns3::ConstantRandomVariable[Constant=41.65]"),
                                // Mean direction: 0..2pi (radians)
                              "MeanDirection", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.283185307]"),
                                // Mean pitch: keep small so UAV mostly horizontal
                              "MeanPitch", StringValue("ns3::UniformRandomVariable[Min=-0.05|Max=0.05]"),
                                // Random components (tuning “spread”)
                                // NOTE: NormalRandomVariable uses "Variance" (not StdDev)
                              "NormalVelocity", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=4.0]"),
                              "NormalDirection", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=1.0]"),
                              "NormalPitch", StringValue("ns3::NormalRandomVariable[Mean=0.0|Variance=0.02]"));

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

    AnimationInterface anim("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_GA/simulasi_dengan_30_node/20/fanet-animation.xml");
    anim.SetMobilityPollInterval(Seconds(0.1));
    anim.EnablePacketMetadata(true);

    uint32_t uavImageId = anim.AddResource("/home/rangga/ns-3-dev/uav.svg");
    for (uint32_t i = 0; i < numUAVs; ++i) {
        anim.UpdateNodeImage(uavNodes.Get(i)->GetId(), uavImageId);
        anim.UpdateNodeSize(uavNodes.Get(i)->GetId(), 50.0, 50.0);
    }

    // ======== GA-ONLY Optimizer (parameter sama) ========
    GAOptimizer optimizer(30, 50, /*mutation*/0.05, /*elitism*/0.10, /*crossover*/0.8,
                          /*conv*/0.001, /*noImprove*/10);

    for (double t = 60.0; t < simulationTime; t += optimizationInterval) {
        Simulator::Schedule(Seconds(t), [&, t, numUAVs]() {
            currentRound++;

            if (rssiMeasurements.size() < 3 || latencyMeasurements.size() < 3 || pdrMeasurements.size() < 3) {
                return;
            }

            // source = numUAVs-1, dest = 0 (sesuai program awal)
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

    flowMonitor->SerializeToXmlFile("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_GA/simulasi_dengan_30_node/20/flow-monitor.xml", true, true);

    FlowMonitorHelper flowHelper; // declare again for classifier
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

    std::ofstream summaryFile("/media/sf_Ubuntu/skenario konstan/hasil_simulasi_AODV_GA/simulasi_dengan_30_node/20/network_summary.xml");
    summaryFile << "<?xml version=\"1.0\"?>\n<NetworkSummary>\n"
                << "  <TotalRounds>" << currentRound << "</TotalRounds>\n"
                << "  <TotalApplicationPackets>" << totalAppPackets << "</TotalApplicationPackets>\n"
                << "</NetworkSummary>\n";
    summaryFile.close();

    Simulator::Destroy();
    return 0;
}
