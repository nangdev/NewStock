import { Ionicons } from '@expo/vector-icons';
import { ROUTE } from 'constants/routes';
import { usePathname, useRouter } from 'expo-router';
import { View, Text, Pressable } from 'react-native';

const menus = [
  { name: '홈', href: ROUTE.HOME, icon: 'home-outline', activeIcon: 'home' },
  {
    name: '뉴스스크랩',
    href: ROUTE.NEWS.SCRAP,
    icon: 'newspaper-outline',
    activeIcon: 'newspaper',
  },
  { name: '뉴스레터', href: ROUTE.NEWSLETTER.CALENDAR, icon: 'mail-outline', activeIcon: 'mail' },
] as const;

export default function CustomFooter() {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <View className="absolute bottom-0 z-10 w-full flex-row border-t border-gray-200 bg-white">
      {menus.map((menu) => {
        const isActive = pathname === menu.href || pathname.startsWith(menu.href + '/');

        return (
          <Pressable
            key={menu.href}
            onPress={() => {
              if (!isActive) router.navigate(menu.href);
            }}
            className="flex-1 items-center justify-center py-2">
            <Ionicons
              name={isActive ? menu.activeIcon : menu.icon}
              size={24}
              color={isActive ? '#724EDB' : '#6b7280'}
            />
            <Text className={isActive ? 'mt-1 text-xs text-primary' : 'mt-1 text-xs text-gray-500'}>
              {menu.name}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}
