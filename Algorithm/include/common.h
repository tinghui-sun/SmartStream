#pragma once
#include <string>

struct AddrInfo
{
	std::string m_ip;
	uint32_t m_port;
};

bool operator == (const AddrInfo& x, const AddrInfo& y)
{
	if (x.m_ip == y.m_ip)
		return x.m_port == y.m_port;
	return false;
}

bool operator < (const AddrInfo& x, const AddrInfo& y)
{
	if (x.m_ip == y.m_ip)
		return x.m_port < y.m_port;
	return x.m_ip < y.m_ip;
}