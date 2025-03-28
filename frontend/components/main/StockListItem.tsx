import React, { useState } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';
import { AntDesign } from '@expo/vector-icons';
import Collapsible from 'react-native-collapsible';

type Props = {
  stockName: string;
  stockCode: string;
  price: number;
  changeRate: number;
  imgUrl: string;
  hojaeIconUrl: string;
};

export default function StockListItem({
  stockName,
  stockCode,
  price,
  changeRate,
  imgUrl,
  hojaeIconUrl,
}: Props) {
  const isPositive = changeRate > 0;
  const [expanded, setExpanded] = useState(false);

  const newsList = [
    { id: 1, title: `${stockName} 관련 뉴스 1`, time: '1시간 전' },
    { id: 2, title: `${stockName} 관련 뉴스 2`, time: '2시간 전' },
    { id: 3, title: `${stockName} 관련 뉴스 3`, time: '2시간 전' },
    { id: 4, title: `${stockName} 관련 뉴스 4`, time: '2시간 전' },
    { id: 5, title: `${stockName} 관련 뉴스 5`, time: '2시간 전' },
  ];

  return (
    <View style={styles.cardContainer}>
      <View style={styles.card}>
        <Image source={{ uri: imgUrl || 'https://via.placeholder.com/48' }} style={styles.logo} />
        <View style={styles.info}>
          <Text style={styles.name}>{stockName}</Text>
          <Text style={styles.code}>{stockCode}</Text>
        </View>
        <View style={styles.priceBlock}>
          <Text style={styles.price}>{isPositive
          ? price.toLocaleString()
          : price.toLocaleString()
          } 원</Text>
          <Text style={[styles.rate, isPositive ? styles.red : styles.blue]}>
            {changeRate.toFixed(2)}%
          </Text>
        </View>
        <TouchableOpacity onPress={() => setExpanded(!expanded)}>
          <AntDesign name={expanded ? 'up' : 'down'} style={styles.toggleButton} />
        </TouchableOpacity>
      </View>

      <Collapsible collapsed={!expanded}>
        <View style={styles.newsSection}>
          {newsList.map((news) => (
            <View key={news.id} style={styles.newsItem}>
              <Image source={{ uri: hojaeIconUrl || 'https://via.placeholder.com/48' }} style={styles.hojaeIcon} />
              <View style={styles.newsRow}>
                <Text style={styles.newsText}>{news.title}</Text>
                <Text style={styles.newsTime}>{news.time}</Text>
              </View>
            </View>
          ))}
        </View>
      </Collapsible>
    </View>
  );
}

const styles = StyleSheet.create({
  cardContainer: {
    backgroundColor: 'white',
    borderRadius: 16,
    marginVertical: 8,
    marginHorizontal: 12,
    shadowColor: '#000',
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 1,
    overflow: 'hidden',
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
  },
  logo: {
    width: 48,
    height: 48,
    borderRadius: 12,
    marginRight: 12,
    backgroundColor: '#eee',
  },
  info: {
    flex: 1,
  },
  name: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  code: {
    fontSize: 12,
    color: '#999',
  },
  priceBlock: {
    alignItems: 'flex-end',
    marginRight: 8,
  },
  price: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  rate: {
    fontSize: 14,
    marginTop: 4,
  },
  red: {
    color: 'red',
  },
  blue: {
    color: 'blue',
  },
  toggleButton: {
    fontSize: 14,
    color: '#888',
    marginLeft: 8,
  },
  newsSection: {
    paddingHorizontal: 16,
    paddingBottom: 12,
    backgroundColor: '#f9f9f9',
  },
  newsItem: {
    marginVertical: 4,
    flexDirection: 'row',
  },
  hojaeIcon: {
    width: 36,
    height: 36,
    borderRadius: 9,
    marginRight: 9,
    backgroundColor: '#eee',
  },
  newsRow: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  newsText: {
    fontSize: 14,
  },
  newsTime: {
    fontSize: 12,
    color: '#888',
  },
});
