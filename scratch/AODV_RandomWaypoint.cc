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

using namespace ns3;

uint32_t currentRound = 0;

NS_LOG_COMPONENT_DEFINE("FANET_AODV_Baseline");

struct FlowStats {
    uint32_t txPackets = 0;
    uint32_t rxPackets = 0;
    double totalDelay = 0.0;
    double totalRssi = 0.0;
    uint32_t rssiCount = 0;
};

Ptr<FlowMonitor> flowMonitor;
std::map<uint32_t, std::ostringstream> nodeDataBuffers;
std::map<std::pair<uint32_t, uint32_t>, FlowStats> flowStatsMap;
std::map<uint64_t, std::pair<double, uint32_t>> packetTxInfo;
std::map<std::pair<uint32_t, uint32_t>, double> rssiMeasurements;
std::map<std::pair<uint32_t, uint32_t>, double> latencyMeasurements;
std::map<std::pair<uint32_t, uint32_t>, double> distanceMeasurements;
std::map<uint64_t, double> packetProcessingStartTimes;
double maxCommunicationRange = 1000.0;

void TransmitPacketCallback(std::string context, Ptr<const Packet> packet, double txPowerDbm) {
    std::string nodeStr = context.substr(10);
    size_t devicePos = nodeStr.find("/Device");
    uint32_t nodeId = std::stoi(nodeStr.substr(0, devicePos));
    
    double currentTime = Simulator::Now().GetSeconds();
    packetTxInfo[packet->GetUid()] = std::make_pair(currentTime, nodeId);
    
    for (auto& entry : flowStatsMap) {
        if (entry.first.first == nodeId) {
            entry.second.txPackets++;
        }
    }
}

void ReceivePacketCallback(std::string context, Ptr<const Packet> packet,
                         uint16_t channelFreqMhz, WifiTxVector txVector,
                         MpduInfo mpduInfo, SignalNoiseDbm signalNoise,
                         uint16_t staId) 
{
    std::string nodeStr = context.substr(10);
    size_t devicePos = nodeStr.find("/Device");
    uint32_t receiverNodeId = std::stoi(nodeStr.substr(0, devicePos));

    Ptr<Packet> copy = packet->Copy();
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    bool isAppPacket = false;
    
    if (copy->RemoveHeader(ipHeader) && ipHeader.GetProtocol() == 17) {
        if (copy->RemoveHeader(udpHeader)) {
            isAppPacket = (udpHeader.GetDestinationPort() == 9);
        }
    }

    double rxTime = Simulator::Now().GetSeconds();

    auto txInfoIt = packetTxInfo.find(packet->GetUid());
    if (txInfoIt == packetTxInfo.end()) return;
    
    double txTime = txInfoIt->second.first;
    uint32_t senderNodeId = txInfoIt->second.second;
    packetTxInfo.erase(txInfoIt);

    if (senderNodeId == receiverNodeId) return;

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

    // Hitung semua komponen delay
    double propagationDelay = (distance > 0) ? (distance / 3e8) : 0.0;
    double transmissionDelay = packet->GetSize() * 8.0 / txVector.GetMode().GetDataRate(txVector);

    // Hitung queue processing delay
    double queueProcessingDelay = 0.0;
    auto procIt = packetProcessingStartTimes.find(packet->GetUid());
    if (procIt != packetProcessingStartTimes.end()) {
        double processingStartTime = procIt->second;
        queueProcessingDelay = rxTime - processingStartTime - propagationDelay - transmissionDelay;
        queueProcessingDelay = std::max(0.0, queueProcessingDelay);
        packetProcessingStartTimes.erase(procIt);
    }

    double perHopDelay = propagationDelay + transmissionDelay + queueProcessingDelay;

    // Update flow statistics
    auto flowKey = std::make_pair(senderNodeId, receiverNodeId);
    FlowStats& flowStats = flowStatsMap[flowKey];
    
    flowStats.rxPackets++;
    flowStats.totalDelay += perHopDelay;
    flowStats.totalRssi += signalNoise.signal;
    flowStats.rssiCount++;

    double pdr = (flowStats.txPackets > 0) ? 
                (static_cast<double>(flowStats.rxPackets) / flowStats.txPackets) * 100.0 : 0.0;
    pdr = std::min(100.0, std::max(0.0, pdr));

    // Format output dengan presisi yang cukup
    std::ostringstream perHopDelayStr, propDelayStr, transDelayStr, queueDelayStr;
    perHopDelayStr << std::fixed << std::setprecision(9) << perHopDelay;
    propDelayStr << std::fixed << std::setprecision(9) << propagationDelay;
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
            
            if (sender == i) {
                totalTx += stats.txPackets;
            }
            
            if (receiver == i) {
                totalRx += stats.rxPackets;
                if (stats.rxPackets > 0) {
                    totalDelay += stats.totalDelay;
                    flowCount++;
                }
            }
        }
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
            
            if (sender == i) {
                totalTx += stats.txPackets;
            }
            
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
        
        std::string filePath = "/media/sf_Ubuntu/hasil_simulasi_AODV/simulasi_dengan_30_node/20/node_" + std::to_string(i) + "_data.xml";
        std::ofstream outFile(filePath);
        if (outFile.is_open()) {
            outFile << nodeDataBuffers[i].str();
            outFile.close();
        }
    }
}

void LogNodePositions(Ptr<Node> node, uint32_t nodeId, double interval) {
    Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
    if (mob) {
        mob->GetPosition();
    }
    Simulator::Schedule(Seconds(interval), &LogNodePositions, node, nodeId, interval);
}

void EnqueuePacketCallback(std::string context, Ptr<const WifiMpdu> mpdu) {
    if (mpdu) {
        double enqueueTime = Simulator::Now().GetSeconds();
        packetProcessingStartTimes[mpdu->GetPacket()->GetUid()] = enqueueTime;
    }
}

int main(int argc, char *argv[]) {
    LogComponentEnable("FANET_AODV_Baseline", LOG_LEVEL_INFO);
    LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    
    uint32_t numUAVs = 100;
    double simulationTime = 300.0;
    double optimizationInterval = 15.0;
    
    NodeContainer uavNodes;
    uavNodes.Create(numUAVs);
    
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss(
        "ns3::RangePropagationLossModel",
        "MaxRange", DoubleValue(maxCommunicationRange)
    );

    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    // phy.Set("TxPowerStart", DoubleValue(20.0));
    // phy.Set("TxPowerEnd", DoubleValue(20.0));
    // phy.Set("RxSensitivity", DoubleValue(-110.0));
    // phy.Set("RxNoiseFigure", DoubleValue(5.0));

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
    // Position allocator untuk area 2D 5000 x 5000
    Ptr<RandomRectanglePositionAllocator> positionAlloc = CreateObject<RandomRectanglePositionAllocator>();
    positionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"));
    positionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=5000.0]"));
    positionAlloc->SetAttribute("Z", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=2000.0]"));


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
    for (uint32_t i = 1; i < uavNodes.GetN(); ++i) {
        nodesWithoutNode0.Add(uavNodes.Get(i));
    }
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
        echoClient.SetAttribute("Interval", TimeValue(Seconds(0.2)));
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
    
    AnimationInterface anim("/media/sf_Ubuntu/hasil_simulasi_AODV/simulasi_dengan_30_node/20/fanet-animation.xml");
    anim.SetMobilityPollInterval(Seconds(0.2));
    anim.EnablePacketMetadata(true);
    
    uint32_t uavImageId = anim.AddResource("/home/rangga/ns-3-dev/uav.svg");
    for (uint32_t i = 0; i < numUAVs; ++i) {
        anim.UpdateNodeImage(uavNodes.Get(i)->GetId(), uavImageId);
        anim.UpdateNodeSize(uavNodes.Get(i)->GetId(), 50.0, 50.0);
    }
    
    for (uint32_t i = 0; i < numUAVs; ++i) {
        Simulator::Schedule(Seconds(0.1), &LogNodePositions, uavNodes.Get(i), i, 1.0);
    }

    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    
    flowMonitor->SerializeToXmlFile("/media/sf_Ubuntu/hasil_simulasi_AODV/simulasi_dengan_30_node/20/flow-monitor.xml", true, true);

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

    std::ofstream summaryFile("/media/sf_Ubuntu/hasil_simulasi_AODV/simulasi_dengan_30_node/20/network_summary.xml");
    summaryFile << "<?xml version=\"1.0\"?>\n<NetworkSummary>\n"
                << "  <TotalRounds>" << currentRound << "</TotalRounds>\n"
                << "  <TotalApplicationPackets>" << totalAppPackets << "</TotalApplicationPackets>\n"
                << "</NetworkSummary>\n";
    summaryFile.close();
    
    Simulator::Destroy();
    
    return 0;
}
