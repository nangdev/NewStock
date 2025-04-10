import { Ionicons } from '@expo/vector-icons';
import { useNotificationDeleteMutation, useNotificationReadMutation } from 'api/notification/query';
import { useAllUserStockListQuery } from 'api/stock/query';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import {
  Modal,
  View,
  Text,
  FlatList,
  TouchableOpacity,
  Pressable,
  Animated,
  Easing,
  Image,
} from 'react-native';
import { NotificationType } from 'types/api/notification';
import { getTimeAgo } from 'utils/date';

type NotificationPopoverProps = {
  visible: boolean;
  onClose: () => void;
  notifications: NotificationType[];
  refetch: () => void;
};

export default function NotificationPopover({
  visible,
  onClose,
  notifications,
  refetch,
}: NotificationPopoverProps) {
  const [showModal, setShowModal] = useState(false);
  const router = useRouter();

  const { mutate: readNotification } = useNotificationReadMutation();
  const { mutate: deleteNotification } = useNotificationDeleteMutation();
  const { data: userStockData, isSuccess: isUserStockSuccess } = useAllUserStockListQuery();

  const translateY = useRef(new Animated.Value(-1000)).current;

  useEffect(() => {
    if (visible) {
      setShowModal(true);
      Animated.timing(translateY, {
        toValue: 0,
        duration: 300,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: true,
      }).start();
    } else {
      Animated.timing(translateY, {
        toValue: -1000,
        duration: 300,
        easing: Easing.in(Easing.ease),
        useNativeDriver: true,
      }).start(() => {
        setShowModal(false);
      });
    }
  }, [visible]);

  const handleRead = (unId: number) => {
    readNotification(
      { unId },
      {
        onSuccess: () => {
          const target = notifications.find((item) => item.unId === unId);
          if (!target) return;
          if (target.newsInfo.newsId) {
            router.navigate(ROUTE.NEWS.DETAIL(target.newsInfo.newsId));
          }
        },
      }
    );
  };

  const handleDelete = (unId: number) => {
    deleteNotification({ unId }, { onSuccess: refetch });
  };

  if (!showModal) return null;

  return (
    <Modal transparent animationType="fade">
      <Pressable onPress={onClose} className="flex-1 bg-black/30">
        <Animated.View
          style={{ transform: [{ translateY }] }}
          className="absolute left-0 right-0 top-0 h-[60%] rounded-b-2xl bg-white p-4 shadow-lg">
          <View className="mb-2 flex-row items-center justify-between border-b border-stroke pb-4">
            <Text className="text-lg font-bold">주요 뉴스 알림</Text>
          </View>

          {notifications.length ? (
            <FlatList
              data={notifications}
              keyExtractor={(item) => item.unId.toString()}
              renderItem={({ item }) => {
                const currentStock = userStockData?.data.stockList.find(
                  (stock) => stock.stockId === item.stockInfo.stockId
                );

                return (
                  <View className="flex-row items-center justify-center gap-2 border-b border-gray-200 py-3">
                    <TouchableOpacity
                      onPress={() => handleRead(item.unId)}
                      className="flex-1 flex-row items-center gap-3 px-2">
                      {!item.isRead ? (
                        <View className="h-2 w-2 rounded-full bg-red-500" />
                      ) : (
                        <View className="h-2 w-2" />
                      )}
                      <View className="gap-2">
                        <View className="flex-row items-center gap-2">
                          <Image
                            source={{ uri: `data:image/png;base64,${currentStock?.imgUrl}` }}
                            style={{
                              width: 20,
                              height: 20,
                              resizeMode: 'contain',
                              borderRadius: 4,
                            }}
                          />
                          <Text
                            className={`font-bold ${item.isRead ? 'text-gray-400' : 'text-text'}`}>
                            {item.stockInfo.stockName}
                          </Text>
                        </View>
                        <Text
                          className={`text-xs ${item.isRead ? 'text-gray-400' : 'text-text'}`}
                          numberOfLines={1}>
                          {item.newsInfo.title}
                        </Text>
                      </View>
                    </TouchableOpacity>
                    <View className="items-center justify-center">
                      <TouchableOpacity
                        onPress={() => handleDelete(item.unId)}
                        className="self-end p-2">
                        <Ionicons name="trash-outline" size={20} color="gray" />
                      </TouchableOpacity>
                      <Text className="text-xs text-gray-400">
                        {getTimeAgo(item.newsInfo.publishedDate)}
                      </Text>
                    </View>
                  </View>
                );
              }}
            />
          ) : (
            <View className="flex-1 items-center justify-center">
              <Image
                source={require('../../assets/image/no_data.png')}
                style={{ width: 50, height: 50, resizeMode: 'contain' }}
              />
              <Text style={{ color: '#8A96A3' }}>뉴스 알림이 없어요</Text>
            </View>
          )}
        </Animated.View>
      </Pressable>
    </Modal>
  );
}
