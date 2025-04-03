import { View, ScrollView, Text } from "react-native";
import { useLocalSearchParams } from "expo-router";
import StockInfoCard from "components/stock/StockInfoCard";
import { useStockDetailInfoQuery } from "api/stock/query";


export default function StockDetail () {
  const { stockId } = useLocalSearchParams();
  const code = Array.isArray(stockId) ? stockId[0] : stockId ?? '';
  const { data, isLoading, isError } = useStockDetailInfoQuery(+stockId);
  
  return (
    <View>
      <Text>stock detail page {stockId}</Text>
      <StockInfoCard 
        stockId={+stockId}
        stockName={data?.data.stockName ?? ''}
        // stockCode={}
        price={data?.data.closingPrice ?? 0}
        changeRate={Number(data?.data.rcPdcp ?? 0)}
        imgUrl={data?.data.stockImage ?? ''}
      />
    </View>
  );
}